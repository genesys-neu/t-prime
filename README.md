# Main Pipeline
To run real time classification use:
```
python3 dstl_run.py [--frequency] [--timeout] [--model_path] [--model_size] [--RMSNorm] [--tensorRT] [--protocols]
```
All arguments are optional. 
--frequency (or -fq) specifies the center frequency the device will be tuned to; the default value is 2.457e9 (WiFi channel 10). 
--timeout (or -t) specifies the time (in seconds) to run before gracefully shutting down; the default is 60 seconds. 
--model_path is self explanatory; the default path is the current directory. 
--model_size specifies if you are using the large "lg" or small transformer "sm". Note that other models may require additional changes. 
--RMSNorm is a flag that specifies to use the RMSNorm block. 
--tensorRT is a flag that specifies to use TensorRT optimization. 
--protocols is a list of the classes to be considered; the default list is ['802_11ax', '802_11b', '802_11n', '802_11g', 'noise'].
An example of a complete command:
```
python dstl_run.py -fq 2.427e9 -t 180 --model_path dstl_transformer/model_cp/model_lg_otaglobal_inf_RMSn_bckg_ft.pt --model_size lg --tensorRT --RMSNorm
```

# Transformer models
All DSTL specific transformer models are in the /dstl_transformer folder.


# Pre-processing
All preprocessing specific code is in the /preprocessing folder.


# Other models
The CNN code is in the /cnn_baseline folder.

All huggingface transformer based code is in the /transformer folder. This was only used for the initial visual transformer testing, and is not present or necessary for the final implementation.
