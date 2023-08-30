# Dataset folder
This is the default folder used to read datasets by training and test scripts provided in this repository.

The complete set of datasets can be found [here](https://northeastern-my.sharepoint.com/:f:/r/personal/sioannidis_northeastern_edu/Documents/DSTL-NU%20OneDrive/Datasets?csf=1&web=1&e=RDViVZ).

## Table of provided datasets
| ID     | Medium     | Tx / Rx          | Mode          | # of examples            | Description                                          | Zip file name   |
|--------|------------|------------------|---------------|--------------------------|------------------------------------------------------|-----------------|
| DS 1.0 | Synthetic  | -                | Single protocol | 2000 for each protocol  | Single packet per file (unbalanced dataset)          | DATASET_1_0.zip |
| DS 1.1 | Synthetic  | -                | Single protocol | 2000 for each protocol  | Multiple packet per file (balanced dataset)     | DATASET_1_1.zip |
| DS 2.0 | Wire       | NI-API           | Multi-protocol | 800 non-overlapping, 3300 overlapping | Wideband spectrogram. Overlapping and non-overlapping | DATASET_2_0.zip | 
| DS 3.0 | OTA        | NI-API / AIR-T   | Single protocol | 200 for each protocol   | Only one protocol transmitted at a time              | DATASET_3_0.zip |
| DS 3.1 | OTA        | NI-API / AIR-T   | Multi-protocol | 400 for each overlap configuration | 6 combinations. 25% and 50% overlap                  | DATASET_3_1.zip |
| DS 3.2 | OTA        | NI-API / AIR-T   | Multi-protocol | 200 for each overlap configuration | 6 combinations. 25% and 50% overlap                  | DATASET_3_2.zip |
| DS 3.3 | OTA        | NI-API / AIR-T   | Multi-protocol | 200 for each overlap configuration | 12 combinations; 25% and 50% overlap                 | DATASET_3_3.zip |

Overview of all datasets (DS) collected in this work. DS 1.0 and DS 1.1 consist of just baseband waveforms (random synthetic channel is applied dynamically at runtime). For OTA and wired collected signals, we use Ettus USRP X310 as the transmitter hardware and [NI-NEU RF Data Recording API (NI-API)](https://github.com/genesys-neu/ni-rf-data-recording-api) to control the USRP.

