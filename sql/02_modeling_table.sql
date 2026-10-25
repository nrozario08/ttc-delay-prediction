-- 02_modeling_table.sql
-- One row per logged incident, with the features the model is allowed to see
-- and the thing it is trying to predict.
--
-- Target: major_delay = 1 when the incident ended up causing 10+ minutes of delay.
--
-- Features are limited to what is known when an incident is first logged:
-- when and where it happened, the line and direction, and the delay code
-- (the reported cause). Left out on purpose:
--   * min_gap  - the gap between trains is only known after the fact, it would leak the answer
--   * vehicle  - just an id number, and 0 for most rows
--   * year     - the split is by time instead, so the model can't lean on the year

-- "How busy was the system just before this incident?" Counts how many other incidents
-- were logged in the hour before, on the same line and across the whole subway. Only
-- earlier logging times are used (not their delay minutes, which may not be known yet).
DROP TABLE IF EXISTS recent_activity;
CREATE TABLE recent_activity AS
WITH stamped AS (
    SELECT
        rowid AS incident_id,
        line,
        CAST(STRFTIME('%s', delay_date || ' ' || time_of_day || ':00') AS INTEGER) AS ts
    FROM delays
)
SELECT
    incident_id,
    COUNT(*) OVER (
        PARTITION BY line ORDER BY ts
        RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
    ) AS line_incidents_prev_hour,
    COUNT(*) OVER (
        ORDER BY ts
        RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
    ) AS system_incidents_prev_hour
FROM stamped;

CREATE INDEX idx_recent_id ON recent_activity (incident_id);


DROP TABLE IF EXISTS model_data;
CREATE TABLE model_data AS
SELECT
    d.rowid                                           AS incident_id,
    d.delay_date,
    d.year,
    d.month,
    d.hour,
    d.weekday,
    d.is_weekend,
    -- rush hour flag so the model gets the idea without having to learn it from 24 hour values
    CASE WHEN d.hour BETWEEN 6 AND 9 OR d.hour BETWEEN 15 AND 18 THEN 1 ELSE 0 END AS is_rush_hour,
    -- stations not in the cleaned station list (yards, whole-line entries, rare spellings) get grouped
    COALESCE(s.station, 'OTHER')                      AS station,
    d.line,
    COALESCE(d.direction, 'NONE')                     AS direction,
    d.code,
    d.category,
    r.line_incidents_prev_hour,
    r.system_incidents_prev_hour,
    d.min_delay,
    CASE WHEN d.min_delay >= 10 THEN 1 ELSE 0 END    AS major_delay
FROM delays d
LEFT JOIN station_dim s    ON s.station = d.station
JOIN recent_activity r     ON r.incident_id = d.rowid;

CREATE INDEX idx_model_year ON model_data (year);
