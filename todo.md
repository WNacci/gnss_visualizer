Please implement these analyses and visualizations. Separate them into self-contained scripts or notebooks as needed.

- [x] The group dynamics notebook does not allow you to aggregate data. It should do so similar to the occupancy heatmap script. It also seems to take a long time to load.
- [x] In fact, all of the notebooks should have the trial aggregation and alignment similar to the occupancy heatmap script. Maybe it can be functionized, but this is also why it's so important to verify config/rotation is correct.
- [x] The site discovery events notebook does not properly display an output.
- [x] Reward site proximity script also fails to display output properly.

- [x] Double check that aggregation orientation is correct. (the data almost looks like a few samples are not rotated/reflected correctly..?)
- [x] Approach to simplify analysis and get some clear, simple metrics: Create radii around each reward site, track sheep count near each site throughout trial to collapse the dimensionality down to 12 or so.
- [x] Maybe show the progression of probability of reward site presence over time?
- [x] path length: separate analysis or tool/script to end timing once sheep find final reward site.
  - measure path length to this event.
  - Should serve as a good metric. Also will help having the 'trial end time metric'
- [x] spatial information metric across time
- [x] leader/follower dynamics: identify what a leader means; and identify who leads across time, if this is consistent or not, etc.
- [x] Self avoiding random walk? Learning of visited locations; reduction of return trips?
- [x] What happens when sites are found or not? How does this influence search?
- [x] Add temporal precision- maybe some visualization of progression over time
- [x] Grouping across time? How consistent is flocking throughout the trials?
- [ ] acquire control reward site configurations for plotting & etc. (I might need to address this outside of the computer)
