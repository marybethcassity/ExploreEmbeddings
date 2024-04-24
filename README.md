## Windows (not GPU enabled)

### Installation on Windows 

**Step 1:** Install Visual Studio 2017 [here](https://download.visualstudio.microsoft.com/download/pr/4035d2dd-2d45-48eb-9104-d4dc7d808a7b/f5675416a31cbf8c29e74d75a1790cf7/vs_community.exe)

Make sure to install Desktop development with C++ by checking the box

![image](https://github.com/marybethcassity/PoseEmbeddingsExplore/assets/70182553/30b20a59-4fbb-418b-b735-5dea12b8bfef)


**Step 2:** Restart your computer


**Step 3:** Install Anaconda distribution [here](https://www.anaconda.com/download)


**Step 4:** Install git 
```
conda install -c anaconda git 
```

**Step 5:** Clone the repository
```
git clone https://github.com/marybethcassity/PoseEmbeddingsExplore.git
```

**Step 6:** Navigate into the directory
```
cd /path/to/PoseEmbeddingsExplore
```

If you cloned the repository into C:\Users\your username, this will be:
```
cd C:\Users\your username\PoseEmbeddingsExplore
```
or
```
cd PoseEmbeddingsExplore
```
if you are currently in C:\Users\your username

**Step 7:** Create the anaconda environment 
```
conda env create --name embeddings -f requirements.yml 
```
(you can replace embeddings with any other environment name you want)


**Step 8:** Activate the environment 
```
conda activate embeddings 
```
(make sure to replace embeddings with your environment name if you came up with your own)


**Step 9:** Run the app 

```
python main.py
```

## WSL (GPU enabled)

## Usage 

**Step 1:** Provide the absolute path to the folder containing csv files from [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) and corresponding mp4 files. For help with DeepLabCut install see [this](https://docs.google.com/document/d/1VsdeL4G_OTTggeyv5SzAn8GRBjdLrDKKMCqUvYcxpRQ/edit?usp=sharing) document. 

**Step 2:** Check the box to determine if you are going to load a previously created plot or create a new one. 

#### If creating a new plot

**Step 3:** Name the plot. This will be the name of the subdirectory containing the plot json and the corresponding files.

**Step 4:** Set the parameters. The default parameters will automatically load. 

**Step 5:** Click the Generate UMAP Embedding button. 

#### If loading a previously created plot

**Step 3:** Select the subdirectory containing the plot json you want to load.
> [!NOTE]
The dropdown menu will contain the names of all of the subdirectorys in the main folder you provided (containing the csv and mp4 files). You do not provide the path to the plot json file in Step 1, you select the subdirectory containing the plot json from the drop down.

**Step 4:** Click the Generate UMAP Embedding button. 

#### Optional


