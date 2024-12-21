# TensorFlow Environment Setup and Project Instructions

## Environment Setup

To set up the environment for this project, follow these steps:

1. **Ensure you have Conda installed**  
   Download and install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Conda environment**  
   Use the provided `environment.yml` file to create the environment:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**  
   After creating the environment, activate it using:
   ```bash
   conda activate rotnet
   ```

4. **Adjust GPU dependencies if necessary**  
   The provided `environment.yml` is configured for CUDA 11.2 and cuDNN 8.1, suitable for an NVIDIA RTX 4060 GPU.  
   If your GPU uses a different CUDA or cuDNN version, adjust the following lines in the `environment.yml` file accordingly:
   ```yaml
   - cudatoolkit=<your_cuda_version>
   - cudnn=<your_cudnn_version>
   ```
   Refer to the [TensorFlow GPU Support Matrix](https://www.tensorflow.org/install/source#gpu) to find compatible versions.

---

## Running the Training Script

To train the model, run the training script located in the `train` folder:
```bash
python train/train_carcass_mult-head_lateset.py
```

- **Output**: The trained model will be saved in the `models` folder.

---

## Testing the Results

After training, you can view the results in the `test` folder. These results showcase the performance of the trained model.

---

## Note on Data Availability

Due to the confidential nature of the data used in this project, we regret that the full experimental replication is not possible. However, our team has successfully presented the results to FRONTMATEC during an on-site meeting.

---

## Contact

For any inquiries about the project or environment setup, please feel free to reach out to our team.

--- 
