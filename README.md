Chest xray image classification using transfer learning with GPU in ml.net

Dataset that use I have found here https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/

Transfer learning was usend on ResnetV2101 model. I have tried few more pretrained mode, but ResnetV2101 gave me best results.

Training was done with GPU.

If you dont have CUDA compatibal graphics card or you dont have CUDA installed on you system, uninstall SciSharp.TensorFlow.Redist-Windows-GPU nuget package and install SciSharp.TensorFlow.Redist. This will use CPU for training, but it will be much slower.

To setup CUDA on you sistem flow instruction from link below

https://github.com/dotnet/machinelearning/blob/master/docs/api-reference/tensorflow-usage.md

Model generated can classify chest xray images with pneumonia and without pneumonia.
