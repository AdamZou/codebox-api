import asyncio
import time
from typing import Callable

import pytest
from codeboxapi import CodeBox
from codeboxapi.schema import CodeBoxFile, CodeBoxOutput
from codeboxapi.utils import mse_prediction

AssertFunctionType = Callable[[CodeBoxOutput, list[CodeBoxFile]], bool]

code_1 = """
import pandas as pd
# Read the CSV file
df = pd.read_csv('iris.csv')

# Save the DataFrame to an Excel file
df.to_excel('iris.xlsx', index=False)
"""


def assert_function_1(_, files):
    return any(".xlsx" in file.name for file in files)


code_2 = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('advertising.csv')

# Split the data into features (X) and target (y)
X = data[['TV']]
y = data['Sales']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

mse
"""
def assert_function_2(output, _):
    return 4.0 <= float(output.content) <= 7.0

code_3="""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the train dataset
train_data = pd.read_csv('train_data.csv')

# Split the dataset into features and target
X = train_data[['TV', 'Radio', 'Newspaper']]
y = train_data['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test dataset
test_data = pd.read_csv('test_data.csv')
predictions = model.predict(scaler.transform(test_data[['TV', 'Radio', 'Newspaper']]))

# Save the predictions to a CSV file
prediction_df = pd.DataFrame({'Sales': predictions})
prediction_df.to_csv('prediction.csv', index=False)
"""

def assert_function_3(output, files):
    mse = mse_prediction(output.content, files)
    print(f"MSE: {mse}")
    return mse <= 7.0

code_4="""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the train dataset
train_data = pd.read_csv('train_data.csv')

# Split the dataset into features and target
X_train = train_data[['TV']]
y_train = train_data['Sales']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test dataset
test_data = pd.read_csv('test_data.csv')
predictions = model.predict(scaler.transform(test_data[['TV']]))

# Save the predictions to a CSV file
prediction_df = pd.DataFrame({'Sales': predictions})
prediction_df.to_csv('prediction.csv', index=False)
"""

code_url_upload = """
import requests
import pandas as pd

def download_file_from_url(url: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_name = 'train.csv'
    with open('./' + file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

download_file_from_url('https://drive.google.com/uc?export=download&id=1sYDeT-I--wlmuzGmiqZmxlTYakvrCIt2')
df = pd.read_csv('train.csv')
print(df.shape)
"""

code_gdrive = """
%pip install gdown

import gdown
import pandas as pd

def download_file_from_google_drive(file_id: str, output_path: str) -> None:
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

file_id = '1sYDeT-I--wlmuzGmiqZmxlTYakvrCIt2'  # Update this to your actual file ID
file_path = 'train.csv'
download_file_from_google_drive(file_id, file_path)

df = pd.read_csv(file_path)
print(df.shape)
"""

def assert_function_url_upload(output, files):
    return '1117957' in output.content and any("train.csv" in file.name for file in files)


# Helper function to build parameters with defaults
def param(code, assert_function, files=[], num_samples=2, local=False, packages=[]):
    return (
        code,
        assert_function,
        files,
        num_samples,
        local,
        packages,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, assert_function, files, num_samples, local, packages",
    [
        # param(
        #     code_1,
        #     assert_function_1,
        #     files=[CodeBoxFile.from_path("examples/assets/iris.csv")],
        # ),
        # param(
        #     code_1,
        #     assert_function_1,
        #     files=[CodeBoxFile.from_path("examples/assets/iris.csv")],
        #     num_samples=1,
        #     local=True,
        #     packages=["pandas", "openpyxl"],
        # ),
        # param(
        #     code_2,
        #     assert_function_2,
        #     files=[CodeBoxFile.from_path("examples/assets/advertising.csv")],
        # ),
        # param(
        #     code_2,
        #     assert_function_2,
        #     files=[CodeBoxFile.from_path("examples/assets/advertising.csv")],
        #     num_samples=10,
        # ),  # For remote CodeBox, the time taken to run 10 samples
        # #   should be around the same as 2 samples (the above case).
        # param(
        #     code_2,
        #     assert_function_2,
        #     files=[CodeBoxFile.from_path("examples/assets/advertising.csv")],
        #     num_samples=1,
        #     local=True,
        #     packages=["pandas", "scikit-learn"],
        # ),
        # param(
        #     code_3,
        #     assert_function_3,
        #     files=[CodeBoxFile.from_path("examples/assets/train_data.csv"), CodeBoxFile.from_path("examples/assets/test_data.csv")],
        # ),
        # param(
        #     code_4,
        #     assert_function_3,
        #     files=[CodeBoxFile.from_path("examples/assets/train_data.csv"), CodeBoxFile.from_path("examples/assets/test_data.csv")],
        # ),
        # param(
        #     code_gdrive,
        #     assert_function_url_upload,
        # ),
        param(
            code_url_upload,
            assert_function_url_upload,
        ),
    ],
)
async def test_boxes_async(
    code: str,
    assert_function: AssertFunctionType,
    files: list[CodeBoxFile],
    num_samples: int,
    local: bool,
    packages: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    codeboxes = [CodeBox(local=local) for _ in range(num_samples)]

    start_time = time.perf_counter()
    tasks = [
        run_async(codebox, code, assert_function, files, packages)
        for codebox in codeboxes
    ]
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    with capsys.disabled():
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    assert all(results), "Failed to run codeboxes"


async def run_async(
    codebox: CodeBox,
    code: str,
    assert_function: AssertFunctionType,
    files: list[CodeBoxFile],
    packages: list[str],
) -> bool:
    try:
        assert await codebox.astart() == "started"

        assert await codebox.astatus() == "running"

        orginal_files = await codebox.alist_files()
        for file in files:
            assert file.content is not None
            await codebox.aupload(file.name, file.content)

        codebox_files = await codebox.alist_files()
        assert set(
            [file.name for file in files] + [file.name for file in orginal_files]
        ) == set([file.name for file in codebox_files])

        assert all(
            [
                package_name in str(await codebox.ainstall(package_name))
                for package_name in packages
            ]
        )

        output: CodeBoxOutput = await codebox.arun(code)
        codebox_files_output = await codebox.alist_files()
        output_files = []
        old_file_names = [file.name for file in codebox_files]
        for file in codebox_files_output:
            if file.name not in old_file_names:
                fileb = await codebox.adownload(file.name)
                if not fileb.content:
                    continue
                output_files.append(fileb)
        print(output_files, codebox_files, codebox_files_output)
        assert assert_function(output, output_files)

    finally:
        assert await codebox.astop() == "stopped"

    return True
