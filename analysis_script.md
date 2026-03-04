# GPS Analysis Script

## Goal description
- Create visualization of GPS data from the files in data_gnss_test
- Include data from all GNSS devices in each set
- Use plotly scatter_map to display the data on a map
- Since the indexes of the files are inconsistent, just pair the most recent files together (highest index)
- Some of the files may be empty; exclude these
- Each device should be its own color, and gets darker over time
- For now we can just plot the first 5 data sets to test
- keep the script as simple as possible

Use source ~/Rice/RobinsonLab/Data/.venv/bin/activate for venv
