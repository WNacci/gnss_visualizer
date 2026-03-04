To identify the reward sites' approximate location, we must do the following in a new marimo script within the scripts directory:

1. Isolate the GNSS data for each reward site
  a. This involves selecting the corresponding GNSS device and recording period for each location. This ID and time information is within the reward_sites.md file. The GNSS data is within the Working_Data_2/gnss_data_field file, the gps_app_simple.py file is an example of inputting and parsing this file.
2. Average the location of each reward site
  a. This will involve selecting the average location from the selected data from each site. A likely tuning step is cutting the period to a later slice of the entire recording, as the GNSS devices likely acquired a higher precision lock later in the recordings.
3. Output a data object relating each reward site (label, field, coordinate) with it's precise location (lat/long)
4. Visualize these location
  a. At the end of the script the locations need to be visualized on the map as in the scripts/gps_app_simple.py script, with the metatdata of the label and field shown on the tooltip.
