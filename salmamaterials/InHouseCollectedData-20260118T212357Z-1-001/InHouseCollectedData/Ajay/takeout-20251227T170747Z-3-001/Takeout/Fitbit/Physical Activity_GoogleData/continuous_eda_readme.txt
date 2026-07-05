Time Series Data Export

The Time Series export provides a detailed timeline of your tracked activity.

Files Included:
----------

continuous_eda_YYYY-MM-DD.csv         - Where YYYY-MM-DD is the starting date for the entries in the file.

Each entry has the following values:

    timestamp                               - Date and time at which the entry was logged.
    eda level real micro siemens            - An estimate of the intra-minutely central point of electrical admittance between two electrodes on the top of the wrist.
    eda slope real micro siemens per minute - Per-minute slope of the EDA level.
    scr detection count                     - The number of SCRs (EDA peaks) detected in the 1 minute integration time.
    leads contact count                     - The number of samples (out of approximately 250) where the cEDA sensor was in contact with the skin.
    minimum i ohms                          - The minimum value from the 25Hz raw i (resistance) cEDA value (in impedance) over the current minute.
    maximum i ohms                          - The maximum value from the 25Hz raw i (resistance) cEDA value (in impedance) over the current minute.
    median i ohms                           - The median value from the 25Hz raw i (resistance) cEDA value (in impedance) over the current minute.
    minimum q ohms                          - The minimum value from the 25Hz raw q (reactance) cEDA value (in impedance) over the current minute.
    maximum q ohms                          - The maximum value from the 25Hz raw q (reactance) cEDA value (in impedance) over the current minute.
    median q ohms                           - The median value from the 25Hz raw q (reactance) cEDA value (in impedance) over the current minute.
    data source                             - The origin or source of this data.
