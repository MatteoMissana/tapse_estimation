import os
import datetime
import pandas as pd
from pydicom.filereader import dcmread
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
from torch.utils.data import DataLoader
from scipy import interpolate

from temporal_pipeline.dataloader.data_prep import SingleFileClipDataset
from temporal_pipeline.models.models import UNet3D
from temporal_pipeline.postprocessing.coordinates_calculation_from_masks import argmax_3d, argmax_3d_for_testing
from temporal_pipeline.utils.plot import visualize_image

class RVCalculator:
    def __init__(self, ed_frame, es_frame, method="triangle"):
        self.ed_free_wall = torch.as_tensor(ed_frame[0])
        self.ed_septum = torch.as_tensor(ed_frame[1])
        self.ed_apex = torch.as_tensor(ed_frame[2])

        self.es_free_wall = torch.as_tensor(es_frame[0])
        self.es_septum = torch.as_tensor(es_frame[1])
        self.es_apex = torch.as_tensor(es_frame[2])

        self.method = method
        self._compute()

    def _compute(self):
        ed_mid = (self.ed_free_wall + self.ed_septum) / 2
        es_mid = (self.es_free_wall + self.es_septum) / 2

        self.tapse_fw = torch.linalg.norm(self.ed_free_wall - self.es_free_wall)
        self.tapse_sep = torch.linalg.norm(self.ed_septum - self.es_septum)

        self.ed_len_fw = torch.linalg.norm(self.ed_free_wall - self.ed_apex, dim=-1)
        self.ed_len_sep = torch.linalg.norm(self.ed_septum - self.ed_apex, dim=-1)
        self.ed_len_mid = torch.linalg.norm(ed_mid - self.ed_apex, dim=-1)

        self.es_len_fw = torch.linalg.norm(self.es_free_wall - self.es_apex, dim=-1)
        self.es_len_sep = torch.linalg.norm(self.es_septum - self.es_apex, dim=-1)
        self.es_len_mid = torch.linalg.norm(es_mid - self.es_apex, dim=-1)

        self.ed_diam = torch.linalg.norm(self.ed_free_wall - self.ed_septum, dim=-1)
        self.es_diam = torch.linalg.norm(self.es_free_wall - self.es_septum, dim=-1)

        if self.method == "triangle":
            ed_s = (self.ed_len_fw + self.ed_len_sep + self.ed_diam) / 2
            self.ed_area = torch.sqrt(
                ed_s
                * (ed_s - self.ed_len_fw)
                * (ed_s - self.ed_len_sep)
                * (ed_s - self.ed_diam)
            )

            es_s = (self.es_len_fw + self.es_len_sep + self.es_diam) / 2
            self.es_area = torch.sqrt(
                es_s
                * (es_s - self.es_len_fw)
                * (es_s - self.es_len_sep)
                * (es_s - self.es_diam)
            )

        elif self.method == "spline":
            self.ed_area = self._spline_area(self.ed_free_wall, self.ed_apex, self.ed_septum, n_points=6)
            self.es_area = self._spline_area(self.es_free_wall, self.es_apex, self.es_septum, n_points= 5)

        else:
            raise ValueError("method must be 'triangle' or 'spline'")

        self.rvfac = (self.ed_area - self.es_area) / self.ed_area * 100

        self.strain_fw = (self.ed_len_fw - self.es_len_fw) / self.ed_len_fw * 100
        self.strain_sep = (self.ed_len_sep - self.es_len_sep) / self.ed_len_sep * 100
        self.strain_mid = (self.ed_len_mid - self.es_len_mid) / self.ed_len_mid * 100
        self.strain_global = (
            (self.ed_len_fw + self.ed_len_sep)
            - (self.es_len_fw + self.es_len_sep)
        ) / (self.ed_len_fw + self.ed_len_sep) * 100

    @staticmethod
    def _spline_area(p1, apex, p2, n_points):
        pts = torch.vstack([p1, apex, p2, p1]).cpu().numpy()
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = torch.from_numpy(np.column_stack((x_new, y_new))).to(dtype=torch.float32, device=p1.device)
        return 0.5 * torch.abs(
            torch.dot(poly[:, 0], torch.roll(poly[:, 1], -1))
            - torch.dot(poly[:, 1], torch.roll(poly[:, 0], -1))
        )




class Predictor:
    def __init__(self, args):
        self.args = args
        
        #set device
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        print('Using device:', self.device)

        #load the model and weights
        self.model = UNet3D(
                    device=self.device,
                    initial_channels=self.args.unet_initial_channels,
                    num_res_units=self.args.unet_res_units,
                )            
        self.model.load_state_dict(torch.load(self.args.model_checkpoints, 
        map_location=self.device)['model_state_dict'])

        self.model.eval()

    def create_prediction_excel(self):
        ''' this function takes as imput the arguments passed with predict_indices.py, and 
        returns a blank pd.dataframe, with a column for each index, and some columns to 
        recognise the acquisition: (Patient id, path to the hdf5 file, and datetime of the acquisition).
        The path is needed for clarity and for the prediction step, that comes after this.'''
        # create an excel file with empty fields for each index
        self.df = pd.DataFrame({
            'path':         pd.Series(dtype='str'),
            'Date':     pd.Series(dtype='str'),
            'Time':     pd.Series(dtype='str'),
            'id':           pd.Series(dtype='str'),
            'tapsefw':      pd.Series(dtype='float64'),
            'tapsesep':     pd.Series(dtype='float64'),
            'rvfac':        pd.Series(dtype='float64'),
            'rvad':         pd.Series(dtype='float64'),
            'rvas':         pd.Series(dtype='float64'),
            'rvldfw':       pd.Series(dtype='float64'),
            'rvldsep':      pd.Series(dtype='float64'),
            'rvlsfw':       pd.Series(dtype='float64'),
            'rvlssep':      pd.Series(dtype='float64'),
            'tadd':         pd.Series(dtype='float64'),
            'tasd':         pd.Series(dtype='float64'),
            'rvldmid':      pd.Series(dtype='float64'),
            'rvlsmid':      pd.Series(dtype='float64'),
            'rvlsffw':      pd.Series(dtype='float64'),
            'rvlsfglobal':  pd.Series(dtype='float64'),
            'rvlsfsep':     pd.Series(dtype='float64'),
            'rvlsfmid':     pd.Series(dtype='float64'),
        })

        dcm_path = os.path.join(self.args.test_set_path, 'test_dicom')

        # initialize the line number, that is increased in each cycle of the loop to put data in the correct line
        line = 1

        for folder in os.listdir(dcm_path):
            if folder.startswith('.'):
                continue 

            folder_path = os.path.join(dcm_path, folder)
            for file in os.listdir(folder_path): # for each file (each one is in a separate folder)

                if file.startswith('.'):
                    continue 

                dcm = os.path.join(folder, file)
                # append the path of the file in the path column 
                self.df.loc[line, "path"] = dcm

                #read the dicom file to get the patient id
                ds = dcmread(os.path.join(dcm_path, dcm))

                # write the patient id on the excel
                self.df.loc[line, "id"] = ds.PatientID

                # write the patient's AcquisitionDateTime (optional to recognize the patient if that's not anonymized)
                dt = str(ds.AcquisitionDateTime)
                self.df.loc[line, "Date"] = f"{dt[0:4]}/{dt[4:6]}/{dt[6:8]}"
                self.df.loc[line, "Time"] = f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}"
                
                #increment the line of the excel
                line = line+1   


    def pan_tompkins_detector(self, plot=False):
        def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0, order=1):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, signal)

        ecg_signal = self.ecg
        fs = self.fs

        # 1. Bandpass filter (5–15 Hz)
        filtered = bandpass_filter(ecg_signal, fs, lowcut=5.0, highcut=15.0)

        # 2. Derivative
        derivative = np.diff(filtered)
        derivative = np.append(derivative, 0)

        # 3. Squaring
        squared = derivative ** 2

        # 4. Moving window integration (~150 ms window)
        window_size = int(0.150 * fs)
        mwa = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        # 5. Peak detection with adaptive thresholding
        distance = int(0.25 * fs)
        threshold = 0.5 * np.max(mwa)
        peaks, _ = find_peaks(mwa, height=threshold, distance=distance)

        r_peaks = []
        search_window = int(0.1 * fs)
        for peak in peaks:
            start = max(peak - search_window, 0)
            end = min(peak + search_window, len(ecg_signal))
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)

        self.r_peaks = np.array(r_peaks)

        if plot:
            time = np.arange(len(ecg_signal)) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(time, ecg_signal, label='Original ECG')
            plt.plot(time[r_peaks], ecg_signal[r_peaks], 'ro', label='Detected R Peaks')
            plt.title('ECG Signal with Detected R Peaks')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def inference(self):
        '''predicts the coordinates of the landmarks in the 
        specified hdf5 file'''

        # load the file in 3d images of 32 frames 
        test_dataset = SingleFileClipDataset(self.file_path, clip_length=self.args.window_len, return_heatmaps=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        coordinates_list=[]
        for images, masks in test_loader:
            # inference
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            if self.args.heatmap_method:
                # extract the maximum activated pixel from the heatmaps
                com_tensor = argmax_3d(outputs, device=self.device)
                coordinates_list.append(com_tensor)
        
        # Stack along dimension 1
        self.coordinates_array = torch.cat(coordinates_list, dim=1)


    def find_es(self):
        '''
        function that finds the es and ed frames from a window of predictions spanning from one r peak to the next one
        args: 
        window: np.ndarray with shape (n, 3, 2) containing the coordinates of the 3 landmarks for each of the frames in an heartbeat

        returns:
        self.es: index of the frame where the fw annular point is the farthest from its ED position (frame 0). 
        The index is relative to the beginning of the window.
        '''
        fw = self.window[:,0]
        distances_from_ed = torch.linalg.norm(fw - fw[0], axis = 1)
        es = distances_from_ed.argmax()
        return es


    def calculate_indices(self):
        self.index_container = []
        for i in range(len(self.beat_start)):
            # extract the window of frames corresponding to one heartbeat
            # so here the problem is that since the model predicts clip_start
            # of n frames, there's the possibility that n-1 frames are left out
            # of the prediction, I need to check if the next r peak is in the 
            # predicted coordinates and act consequently.
            # if it is then I consider a heartbeat the window of frames
            # between this beat and the next one. If not then the beat is valid if 
            # there are at least 20 frames after my r peak. and I use those
            # frames as my window.
            if self.beat_start[i] > self.coordinates_array.shape[1]: # if the current R peak is out of the predicted coordinates
                print("skip heart cycle")
                continue
            elif i == len(self.beat_start)-1: # if it's the last r peak and it's inside the predictions
                if self.coordinates_array.shape[1] - self.beat_start[i] <= 20: # if no more than 20 frames after it skip the heartcycle
                    print("skip heart cycle")
                    continue
                else: # more than 20 frames after it
                    self.window = self.coordinates_array[0, self.beat_start[i]:]
            else: #if not last heartbeat and next one is not out of the predictions
                if self.beat_start[i+1] < self.coordinates_array.shape[1]: # if the next one is inside the predictedc interval
                    self.window = self.coordinates_array[0, self.beat_start[i]:self.beat_start[i+1]]

                else: #if not last heartbeat and next one is out of the predictions
                    if self.coordinates_array.shape[1] - self.beat_start[i] <= 20:
                        print("skip heart cycle")
                        continue
                    else: 
                        self.window = self.coordinates_array[0, self.beat_start[i]:]

            
            # in the window, find the index of the ED and ES frame
            ed_idx = 0 # since the window always starts from R-peak
            es_idx = self.find_es()

            # etract these frames
            self.ed = self.window[ed_idx]            
            self.es = self.window[es_idx]
            
            # rescale the coordinates based on pixelsize
            self.ed[:,0] *= self.pixelsize[0]
            self.ed[:,1] *= self.pixelsize[1]
            self.es[:,0] *= self.pixelsize[0]
            self.es[:,1] *= self.pixelsize[1]

            calculator = RVCalculator(
                ed_frame=self.ed,
                es_frame=self.es,
                method='spline')
            
            self.index_container.append(torch.tensor([
                calculator.tapse_fw * 1000, #0
                calculator.tapse_sep * 1000, #1
                calculator.rvfac, #2
                calculator.ed_area * 1e4,#3
                calculator.es_area * 1e4,#4
                calculator.ed_len_fw * 1000, #5
                calculator.ed_len_sep * 1000,#6
                calculator.es_len_fw * 1000, #7
                calculator.es_len_sep * 1000, #8
                calculator.ed_diam * 1000,#9
                calculator.es_diam * 1000, #10
                calculator.ed_len_mid * 1000, #11
                calculator.es_len_mid * 1000, #12
                calculator.strain_fw, #13
                calculator.strain_global,#14 
                calculator.strain_sep,  #15
                calculator.strain_mid, #16
            ]))

        #create array with all the predicted indices
        self.index_array = torch.stack(self.index_container, dim=0) # torch.Size([N_beats, 17])
        

    def write_df(self, path):
        '''
        Writes the mean of the calculated indices across all heart cycles for the current file
        into the corresponding row of the dataframe.

        Args:
            path (str): The path identifier for the current file, used to locate the correct row in the dataframe.

        The indices are averaged across heart cycles and assigned to the dataframe columns in the following order:
        - 'tapsefw': Mean TAPSE free wall (mm)
        - 'tapsesep': Mean TAPSE septum (mm)
        - 'rvfac': Mean RV fractional area change (%)
        - 'rvad': Mean RV area at end-diastole (cm²)
        - 'rvas': Mean RV area at end-systole (cm²)
        - 'rvldfw': Mean RV length diastolic free wall (mm)
        - 'rvldsep': Mean RV length diastolic septum (mm)
        - 'rvlsfw': Mean RV length systolic free wall (mm)
        - 'rvlssep': Mean RV length systolic septum (mm)
        - 'tadd': Mean TAPSE diameter diastolic (mm)
        - 'tasd': Mean TAPSE diameter systolic (mm)
        - 'rvldmid': Mean RV length diastolic mid (mm)
        - 'rvlsmid': Mean RV length systolic mid (mm)
        - 'rvlsffw': Mean RV longitudinal strain free wall (%)
        - 'rvlsfglobal': Mean RV longitudinal strain global (%)
        - 'rvlsfsep': Mean RV longitudinal strain septum (%)
        - 'rvlsfmid': Mean RV longitudinal strain mid (%)
        '''
        # Find the row index in the dataframe corresponding to the current path
        row_idx = self.df[self.df['path'] == path].index[0]
        
        # Compute the mean across heart cycles (dim=0)
        mean_indices = self.index_array.mean(dim=0).cpu().numpy()
        
        # Assign the mean values to the dataframe columns
        self.df.loc[row_idx, 'tapsefw'] = mean_indices[0]
        self.df.loc[row_idx, 'tapsesep'] = mean_indices[1]
        self.df.loc[row_idx, 'rvfac'] = mean_indices[2]
        self.df.loc[row_idx, 'rvad'] = mean_indices[3]
        self.df.loc[row_idx, 'rvas'] = mean_indices[4]
        self.df.loc[row_idx, 'rvldfw'] = mean_indices[5]
        self.df.loc[row_idx, 'rvldsep'] = mean_indices[6]
        self.df.loc[row_idx, 'rvlsfw'] = mean_indices[7]
        self.df.loc[row_idx, 'rvlssep'] = mean_indices[8]
        self.df.loc[row_idx, 'tadd'] = mean_indices[9]
        self.df.loc[row_idx, 'tasd'] = mean_indices[10]
        self.df.loc[row_idx, 'rvldmid'] = mean_indices[11]
        self.df.loc[row_idx, 'rvlsmid'] = mean_indices[12]
        self.df.loc[row_idx, 'rvlsffw'] = mean_indices[13]
        self.df.loc[row_idx, 'rvlsfglobal'] = mean_indices[14]
        self.df.loc[row_idx, 'rvlsfsep'] = mean_indices[15]
        self.df.loc[row_idx, 'rvlsfmid'] = mean_indices[16]


    def predict(self):
        '''
        # TODO
        '''

        # ensure path column exists
        if "path" not in self.df.columns:
            ValueError("Excel file must contain a 'path' column.")

        # extract the file paths
        paths = self.df["path"].dropna().tolist()

        for path in paths:
            # compose file path
            self.file_path = os.path.join(
                self.args.test_set_path,
                'test_hdf5', 
                str(path) + "_interpolated.h5" # TODO: exception if the file does not have interpolated in the name
                )

            print(f"Processing: {self.file_path}")

            # Load ultrasound + ECG data
            with h5py.File(self.file_path, 'r') as f:
                self.images = f['tissue']['data'][()]
                images_times = f['tissue']['times'][()]
                self.ecg = f['ecg']['ecg_data'][()]
                ecg_times = f['ecg']['ecg_times'][()]
                self.pixelsize = f['tissue']['pixelsize'][()]

            self.fs = 1 / (ecg_times[1] - ecg_times[0])  # ECG sampling frequency
            dt = images_times[1] - images_times[0]  # Image frame interval

            # Detect R-peaks in ECG
            self.pan_tompkins_detector()

            # for each R-peak, extract the closest frame
            self.beat_start = [np.argmin(np.abs(images_times - ecg_times[r])) for r in self.r_peaks]

            # run the inference to get the coordinates_array
            self.inference()

            # calculate the indices from the predicted coordinates
            self.calculate_indices()
            
            # put the predicted indices in the dataframe and return it
            self.write_df(path)

        #save dataframe in excel
        self.df.to_excel(self.args.excel_path, index=False)

    def compute_coordinates_annotations(self, file_path):
        '''predicts the coordinates of the landmarks in the 
        specified hdf5 file, and returns them together with the manual annotations.
        Very useful for testing'''
        # load the file in 3d images of 32 frames 
        test_dataset = SingleFileClipDataset(
            file_path, 
            clip_length=self.args.window_len, 
            return_heatmaps=False,
            load_from_annotations = True,
            )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False
            )

        coordinates_list=[]
        maxima_list = []
        gt_list = []

        for images, masks in test_loader:
            # inference
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            if self.args.heatmap_method:
                # extract the maximum activated pixel from the heatmaps
                com_tensor, maxima = argmax_3d_for_testing(outputs, device=self.device)
                # print(com_tensor.shape)

                # # Extract the first slice of the first image in the batch
                # img_slice = images[0, 0, 0].cpu().numpy()

                # # Extract (x, y) coordinates for all landmarks
                # pts = com_tensor[0, 0, :].cpu().numpy() 

                # visualize_image(img_slice, points=pts)

                coordinates_list.append(com_tensor)
                maxima_list.append(maxima.permute(0,2,1))
                gt_list.append(masks)

        
        # Stack along dimension 1
        self.coordinates_array = torch.cat(coordinates_list, dim=1)
        self.maxima_array = torch.cat(maxima_list, dim=1)
        self.gt_array = torch.cat(gt_list, dim=1)









