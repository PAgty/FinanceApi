# Use an official Python runtime as a parent image
FROM continuumio/anaconda3:5.0.0

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN conda install -c conda-forge fbprophet
RUN conda install gxx_linux-64
RUN pip install pyramid-arima
# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "api.py"]