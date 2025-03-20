# MLflow Model Packaging Tutorial

This tutorial describes how to package a ML model as an MLflow model, with explanations and code examples for a simple PyTorch-based model.

It does not cover how to train a model using a ML framework, such as PyTorch, and assumes a model has already been trained and saved to the local filesystem.

Full MLflow documentation for packaging a model can be found [here](https://www.mlflow.org/docs/latest/model/python_model). There is a newer, alternate MLflow model packaging process called ["Models From Code"](https://www.mlflow.org/docs/latest/model/models-from-code/), which is not covered by this tutorial (a future tutorial may cover this).

## Conceptual Overview

MLflow provides a standard packaging format for ML models. MLflow model packages can be created for models that have been trained from nearly any ML model training framework, such as PyTorch, scikit-learn, TensorFlow, and many more.

The key steps to packaging an MLflow model are:

1. **Define an MLflow Model Class:** You will create a class that inherits from the `mlflow.pyfunc.PythonModel` class. Your class will encapsulate all the logic needed to load your PyTorch model and invoke a prediction operation using the underlying framework model.  
2. **Specify Dependencies:** You will create a `requirements.txt` file to list all the Python packages your model needs (like `torch`, 
`numpy`, etc.). MLflow will use this to create a reproducible Python environment when the model is invoked.
3. **Define a script to create the MLflow model package:** You will create a `package.py` Python script that calls MLflow's `mlflow.pyfunc.save_model()` function to package all required model code, data, and metadata into a directory structure that follows MLflow's packaging format. This directory will constitute the "MLflow model".

## Step-by-Step Instructions with Example Code

These steps assume you have already:
* Trained a model and saved it using your ML framework of choice. For example, if using PyTorch, you will have a `model.pt` file.  For the purposes of this tutorial, an example `model.pt` file can be created by running [`train.py`](./train.py) after setting up your Python environment (step #3, below). 
* Source code that defines your model architecture and supports runing a forward pass on your PyTorch model.

For this example, we will also assume there is an auxiliary file, `labels.txt` that is needed to perform a transformation on the output after inference.  

### 1. Setup

Setup steps will depend upon how you intend to use this tutorial:

1. If you simply want to follow along with the example code and scripts provided in this tutorial repository, clone the reposistory:
```sh
git clone https://github.com/atolopko-czi/mlflow-example.git 
cd mlflow-example
```

2. If you want to adapt this tutorial repository to package a model that is being maintained in a separate repository, you can clone this repository and rename accordingly:
```sh
git clone https://github.com/atolopko-czi/mlflow-example.git my-mflow-model
cd my-mlflow-model
```

3. If you want to maintain MLflow model packaging functionality in a new, separate repository from the model repository, and not using this tutorial repo as a starting point, create a new project directory:
```sh
# Create a new directory and navigate into it
mkdir my-mlflow-model && cd my-mlflow-model
git init  # optionally
```
This is particularly useful if the model provides a pip-installable package for running inference.

4. If you want to add MLflow model packaging functionality to an _existing_ model repository that you own (or can update), clone the model's repo and `cd` into it (replacing the placeholder variables, below):
```sh
git clone https://github.com/{OWNER}/{EXISTING_MODEL_REPO}.git
cd {EXISTING_MODEL_REPO}.git
```

Next, create and activate a Python virtual environment :
```sh
# Set up a Python virtual environment
python -m venv .venv
source .venv/bin/activate
```

### 2. Organize all model code and data into subdirectories

```sh
mkdir model_code
mkdir model_data
```

Copy any Python modules that are used to define the model architecture and to run a forward pass on the PyTorch model into `model_code`. See [`model_code/pytorch_model.py`](./model_code/pytorch_model.py) as an example. Note that if you are adding MLflow packaging support to an existing model repository that already includes the model's _inference_ code (setup option #4, above), you can skip copying source code to the `model_code` directory and can instead provide paths to this code in step 5, below.

Copy the PyTorch model weights file (usually `model.pt`) into `model_data`, along with any other auxiliary files that may be needed to help process the inputs to or outputs from the model. As an example, this tutorial includes a `labels.txt` file that is used to transform the model's numeric output to text labels. Note that if you are adding MLflow packaging support to an existing model repository that already includes the model's _training_ code (setup option #4, above), you can simply update the training code to save the model weights directly to the `model_data` directory (the example [`train.py`](./train.py) script does this).


### 3. Create `requirements.txt`

This `requirements.txt` file lists the Python dependencies needed to load and execute the PyTorch model. For example:

```
torch==2.6.0
mlflow==2.20.0
pandas==2.2.3
```

It is best to use concrete versions to avoid dependency issues.

Install the requirements to your Python environment:
```sh
pip install -r requirements.txt
```

Note that if you are adding MLflow packaging support to an existing model repository that already includes the model's _inference_ code (setup option #4, above), you may already have a `requirements.txt` file. In that case, you can choose to create an MLflow-specific requirements file like `requirements-mlflow.txt` to keep the inference-time requirements separate from training-time requirements, if these in fact differ.


### 4. Create `model_code/mlflow_model.py`

This file contains the code for loading and using your PyTorch model. See [`mlflow_model.py`](./mlflow_model.py) for a full example.

* The `predict()` method is the only required method. It should invoke the PyTorch model with the `model_input` argument. Python type hints may be used to annotate the input arguments and return type of the method, which MLflow will later use to infer the input and output signatures of the model (see [docs](https://www.mlflow.org/docs/latest/model/python_model#type-hint-usage-in-pythonmodel)).
* You may optionally define a `load_context()` method to perform any expensive, one-time initialization operations before the MLflow model is used for inference. Typically, `load_context()` will load the PyTorch model object and, optionally, read in the contents of any auxiliary data files (MLflow "artifacts"). The model object and auxiliary data can be stored in object instance variables (e.g. `self.model`, etc.). The instance variables can then be used in the `predict()` method. Note that MLflow will call `load_context()` once before any calls are made `predict()`, allowing for any expensive initiatlization operations to be performed just once for repeated `predict()` calls.
* You may optionally define any other methods you need in this class. A good pattern is to define `_preprocess()` and `_postprocess()`  methods, if the inputs and outputs of the PyTorch model will differ from the inputs and outputs that that MLflow model will expect and produce, respectively. You will need to explicitly call this methods from `predict()`, as needed.


### 5. Create `package.py` 

This script brings everything together and saves the MLflow model. The only requirement is that calls `mlflow.pync.save_model()`, which typically looks like:

```py
    mlflow.pyfunc.save_model(
        # The path where the MLflow model will be saved
        path="mlflow_model",
        # An instance of MLflow model class defined in `mlflow_model.py`
        python_model=MyMLflowModel(), 
        # A dictionary that maps logical artifact names to local files, including the PyTorch model weights.
        # These artifact files can be read by the MyMLflowModel() class.
        artifacts={
            "pytorch_model": "model_data/model.pt",
            # Example of adding an auxiliary file:
            "labels_file": "model_data/labels.txt",
        },          
        # The Python dependencies required to run the model
        pip_requirements="requirements.txt",
        # A list of paths to Python directories and/or files that are needed in Python environment to load and run the model
        code_paths=["model_code"],
        # An example input data object that can be used to infer the model's input and output schema. The input example
        # will be stored with the MLflow model so that users of the model can understand the model's expected input format.
        input_example=pd.DataFrame(np.random.rand(3,5))
    )
```

A full example can be found in [package.py](./package.py).

Note that if you are adding MLflow packaging support to an existing model repository that already includes the model's _inference_ code (setup option #4, above), you can add additional Python package (directory) or Python module (file) paths to the `code_paths` argument. This would allow you to avoid copying code into the `model_code` directory.

MLflow documentation for the `save_model()` method is available [here](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.save_model).

Note that MLflow provides both `save_model()` and `log_model()` methods. The `log_model()` method stores the model in an "experiment tracking" server, which is not assumed to be available for this tutorial. So instead, the `save_model()` method is used, which simply saves the model to the local filesystem.

After this step, your directory should look like this:

```
.
├── model_code
│   ├── mlflow_model.py
│   └── pytorch_model.py
├── model_data
│   ├── labels.txt
│   └── model.pt
├── package.py
└── requirements.txt
```


### 6. Run the model packaging script

```
python package.py
```

After running `package.py`, you should have a directory called `mlflow_model` with a structure similar to this:

```
mlflow_model/
├── MLmodel
├── artifacts
│   ├── labels.txt
│   └── model.pt
├── code
│   └── model_code
│       ├── __init__.py
│       ├── mlflow_model.py
│       └── pytorch_model.py
├── conda.yaml
├── input_example.json
├── python_env.yaml
├── python_model.pkl
├── requirements.txt
└── serving_input_example.json
```

### 7. Load & Test Your MLflow Model

Load the saved MLflow model and perform inference:

```py
import mlflow

model = mlflow.pyfunc.load_model("mlflow_model")
sample_input = np.random.randn(3, 5)
print(model.predict(sample_input))
```

Debug `model_code/mlflow_model.py`, and `package.py` as needed. 

A full example for testing the MLflow model prediction is in [predict.py](./predict.py). Note this example provides two methods of using the MLflow model, one for running the model in the current virtual environment, and one for the running the model in a separate, isolated virtual environment, specified via `--local` and `--isolated` option flags, respectively. In the latter case, MLflow will create a temporary virtual environment, install all the required dependencies, and invoke the model by passing it a file containing a seralized version of the input data. Testing the MLflow model invocation in an isolated Python environment can expose packaging-related errors that may be missed when running the model in the local Python environment (e.g missing package dependencies or inference-time source code).

