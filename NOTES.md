2018-08-27
* add global args to allow modules to use GPU
* minor organisation with .gitignore in downloaded model folders
* debug: grid formation ignores GPU in Yolo-v3
* issue: Yolo-v3 now init without loading weights, so it is possible to 
  send the model to GPU, only to make trouble when later trying to load
  parameters from file.
