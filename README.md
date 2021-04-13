# Activity Recognition Modelling
## 1. Project Description
### A. Overall goal  
* The project is about building and evaluating [Activity Recognition](https://en.wikipedia.org/wiki/Activity_recognition/) Models from triaxial inertial sensors data carried by objects (cattle in this case).  
* The Python GUI project is developed as the programming part for fulfilling the thesis *“Evaluating the Impact of Sampling on Activity Monitoring in Dairy Cattle”* in [MSc. Software Systems Science program](https://www.uni-bamberg.de/ma-isosysc/) at the [Chair of Mobile Systems](https://www.uni-bamberg.de/mobi/) - the University of Bamberg  
* This is also a technical part of the FutureIoT/Rindertracking project - https://www.futureiot.de/portfolio/rindertracking/  
* Publication: L. Schmeling, G. Elmamooz, P. T. Hoang, A. Kozar, D. Nicklas, M. Suenkel, S. Thurner, and E. Rauch (2020). *Sensor-based Monitoring of Lying Behaviour in Dairy Cows on Pasture*. Computers and Electronics in Agriculture (**in revision**).  

### B. Problem Context
* Most activity tracking apps installed on wearable devices are relying on sensor data for the purpose of correctly predicting the carrier's activity. The common problem of these devices is the short battery duration, which is mainly due to the high frequency of sensor data updates and computing expense. Practically, there is a trade-off between the prediction accuracy and the high computing and frequency of data update, not to mention data redundancy.  
* This project is to find the answer to the question of how often and how much data a model should consume while remaining significant accuracy in predicting a particular set of cattle activities. This is more specifically to find the most contributing sensor axes and at what sampling rate, and how large a window of sampled data should be used.  
* The project, however, can be used to build models in recognizing human activities, too.  

### C. Development Language and running Environment
* The project is developed with Python 3.8 (64 bit), and **WindowOS** is the recommended running environment.  The GUI is distorted on MacOS as the tkinter library is not well supported by the system. The author is planning to convert this project into the web version.

### D. The project workflow
After researching the previous studies, the workflow is implemented as shown below
<p align="center">
  <img src="references/work_flow.png"  title="hover text">
</p>

### E. Input data source
* The sensor data come from the Bosch BNO055 device which contains inertial sensors originally sampled at 10Hz.   
* The application supports data importing from either database or csv files. The DBMS supported in the project is PostgreSQL version 9.5.17.  
* The two input data tables/csv files called **Train_Valid_Test** and **Monitoring** must contain at least 9 columns (i.e., cattle_id, label, gx, gy, gz, ax, ay, az, timestamp) with the data types are as follow:

| Column name  | Data type | Explanation |
| ------------- | ------------- | ------------- |
| cattle_id | text | ID number of the cow (e.g., DE1234)  |
| label | text | Labelled activity of cows (e.g., Liegen)  |
| gx | double precision | Gyroscope x axis signal value  |
| gy | double precision | Gyroscope y axis signal value  |
| gz | double precision | Gyroscope z axis signal value  |
| ax | double precision | Accelerometer x axis signal value  |
| ay | double precision | Accelerometer y axis signal value  |
| az | double precision | Accelerometer z axis signal value  |
| timestamp | bigint | Timestamp in Unix Epoch time format  |

With these initiatives, it is possible to train a model from mixed data of multiple cattles by storing sensor data of difference cattles in the **Train_Valid_Test** table (distinguished by cattle_id)  

### F. Repository Structure
| Folder/File  | Purpose |
| ------------- | ------------- |
| csv_out | Folder containing csv output files / experiment result for each run |
| src | Main source code |
| src /db_credentials.ini | A template storing credentials and input tables/csv files for input data importing |
| datasets | Folder contain sample input datasets for some cattle |
