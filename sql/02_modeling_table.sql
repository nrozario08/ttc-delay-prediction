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
    d.min_delay,
    CASE WHEN d.min_delay >= 10 THEN 1 ELSE 0 END    AS major_delay
FROM delays d
LEFT JOIN station_dim s ON s.station = d.station;

CREATE INDEX idx_model_year ON model_data (year);
