CREATE TABLE clip_responses (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    clip_id VARCHAR(255) NOT NULL,
    sequence_id INT NOT NULL,
    position_in_sequence INT NOT NULL,
    
    -- Timestamps
    clip_start_time TIMESTAMP NOT NULL,
    clip_end_time TIMESTAMP NOT NULL,
    
    -- Survey Responses
    p_valence FLOAT,
    p_arousal FLOAT,
    i1_valence FLOAT,
    i1_arousal FLOAT,
    
    -- Physiological Columns (Pending Fitbit Sync)
    i2_valence FLOAT DEFAULT NULL,
    i2_arousal FLOAT DEFAULT NULL,
    hr_mean FLOAT DEFAULT NULL,
    rmssd FLOAT DEFAULT NULL,
    
    -- Computed Columns
    e_quadrant VARCHAR(5),
    p_quadrant VARCHAR(5),
    i1_quadrant VARCHAR(5),
    switching_flag BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
