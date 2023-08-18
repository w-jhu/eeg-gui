# Usage
Streaming from IEEG:
<pre>
python eeg_gui.py -u your_username -p your_password data_info.csv --delay
</pre>
Directly from local folder containing EDF files:
<pre>
python eeg_gui.py folder_name --delay
</pre>
(delay defaults to 0.5)
# Results Interpretation
| Column Name        | Value | Meaning                   |
|--------------------|-------|---------------------------|
| Any                |-1     | Incomplete input          |
| sleep_state_X      | 0     | Awake                     |
|                    | 1     | N1 (Light Sleep Stage)    |
|                    | 2     | N2 (Intermediate Stage)   |
|                    | 3     | N3 (Deep Sleep Stage)     |
|                    | 4     | REM (Rapid Eye Movement)  |
|                    | 5     | Undetermined              |
| prediction_X       | 0     | No seizure detected       |
|                    | 1     | Seizure detected          |
| start_X            | NaN   | No seizure detected       |
| end_X              | NaN   | No seizure detected       |

