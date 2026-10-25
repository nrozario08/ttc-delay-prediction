-- 01_clean.sql
-- Turns delays_raw into a tidy `delays` table.
--
-- What gets fixed:
--   * the Line column has dozens of spellings (YU, YUS, LINE 1, YU / BD ...)
--   * station names carry the line in them ("KENNEDY BD STATION") and are
--     spelled differently from year to year
--   * exact duplicate rows
--   * delay codes get joined to their descriptions and rolled up into a category

DROP TABLE IF EXISTS station_alias;
CREATE TABLE station_alias (
    messy TEXT PRIMARY KEY,
    clean TEXT NOT NULL
);

-- a few names that the simple suffix stripping below can't fix on its own
INSERT INTO station_alias (messy, clean) VALUES
    ('VAUGHAN MC',        'VAUGHAN METROPOLITAN CENTRE'),
    ('VMC',               'VAUGHAN METROPOLITAN CENTRE'),
    ('NORTH YORK CTR',    'NORTH YORK CENTRE'),
    ('QUEENS PARK',       'QUEEN''S PARK'),
    ('HWY 407',           'HIGHWAY 407');


DROP TABLE IF EXISTS delays;
CREATE TABLE delays AS
WITH numbered AS (
    -- number identical rows so only the first copy is kept
    SELECT
        r.*,
        ROW_NUMBER() OVER (
            PARTITION BY Date, Time, Station, Code, "Min Delay", "Min Gap", Bound, Line, Vehicle
            ORDER BY rowid
        ) AS copy_no
    FROM delays_raw r
    WHERE Date IS NOT NULL
      AND Time LIKE '__:__'
),
stripped AS (
    SELECT
        n.*,
        -- drop the word STATION and tidy up spaces
        -- (some names were cut off at 22 characters, e.g. "PIONEER VILLAGE STATIO")
        TRIM(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(Station), ' STATION', ''), ' STATIO', ''), ' STATI', ''), '  ', ' ')) AS st1
    FROM numbered n
    WHERE copy_no = 1
),
stations AS (
    SELECT
        s.*,
        -- remove a trailing line tag like " BD" or " YUS"
        CASE
            WHEN st1 LIKE '% BD'  THEN SUBSTR(st1, 1, LENGTH(st1) - 3)
            WHEN st1 LIKE '% YUS' THEN SUBSTR(st1, 1, LENGTH(st1) - 4)
            WHEN st1 LIKE '% YU'  THEN SUBSTR(st1, 1, LENGTH(st1) - 3)
            WHEN st1 LIKE '% SRT' THEN SUBSTR(st1, 1, LENGTH(st1) - 4)
            WHEN st1 LIKE '% SHP' THEN SUBSTR(st1, 1, LENGTH(st1) - 4)
            ELSE st1
        END AS st2
    FROM stripped s
)
SELECT
    DATE(s.Date)                                    AS delay_date,
    CAST(STRFTIME('%Y', s.Date) AS INTEGER)         AS year,
    CAST(STRFTIME('%m', s.Date) AS INTEGER)         AS month,
    CAST(SUBSTR(s.Time, 1, 2) AS INTEGER)           AS hour,
    -- recompute the weekday from the date instead of trusting the Day column
    CASE STRFTIME('%w', s.Date)
        WHEN '0' THEN 'Sunday'    WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'   WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'  WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END                                             AS weekday,
    CASE WHEN STRFTIME('%w', s.Date) IN ('0', '6') THEN 1 ELSE 0 END AS is_weekend,
    s.Time                                          AS time_of_day,
    COALESCE(a.clean, s.st2)                        AS station,
    CASE
        WHEN UPPER(TRIM(s.Line)) IN ('YU', 'YUS', 'LINE 1', 'YU LINE')
            THEN 'Line 1 Yonge-University'
        WHEN UPPER(TRIM(s.Line)) IN ('BD', 'B/D', 'LINE 2', 'BD LINE 2', 'BD LINE')
            THEN 'Line 2 Bloor-Danforth'
        WHEN UPPER(TRIM(s.Line)) IN ('SHP', 'SHEP', 'LINE 4')
            THEN 'Line 4 Sheppard'
        WHEN UPPER(TRIM(s.Line)) IN ('SRT', 'LINE 3')
            THEN 'Line 3 Scarborough RT'
        WHEN UPPER(s.Line) LIKE '%YU%' AND UPPER(s.Line) LIKE '%BD%'
            THEN 'Multiple lines'
        ELSE 'Unknown'
    END                                             AS line,
    -- direction only means something for N/S/E/W, anything else is NULL
    CASE WHEN UPPER(s.Bound) IN ('N', 'S', 'E', 'W') THEN UPPER(s.Bound) END AS direction,
    s.Code                                          AS code,
    COALESCE(c.description, 'UNKNOWN CODE')         AS code_description,
    CASE SUBSTR(s.Code, 1, 1)
        WHEN 'E' THEN 'Vehicle / equipment'
        WHEN 'P' THEN 'Infrastructure (signals, track, power)'
        WHEN 'S' THEN 'Security and disorderly passengers'
        WHEN 'T' THEN 'Operator / transportation'
        WHEN 'M' THEN 'Medical, weather and miscellaneous'
        ELSE 'Other'
    END                                             AS category,
    CAST(s."Min Delay" AS INTEGER)                  AS min_delay,
    CAST(s."Min Gap" AS INTEGER)                    AS min_gap,
    CASE WHEN CAST(s."Min Delay" AS INTEGER) > 0 THEN 1 ELSE 0 END AS caused_delay,
    s.Vehicle                                       AS vehicle
FROM stations s
LEFT JOIN station_alias a ON a.messy = s.st2
LEFT JOIN delay_codes c   ON c.code = UPPER(TRIM(s.Code));

-- Station dimension. The Station column also holds things like "YONGE-UNIVERSITY
-- AND B..." or yard names, so only keep names that look like real stations and show
-- up often enough that a typo is unlikely.
DROP TABLE IF EXISTS station_dim;
CREATE TABLE station_dim AS
SELECT station, COUNT(*) AS total_incidents
FROM delays
WHERE station NOT LIKE '%LINE%'
  AND station NOT LIKE '%YARD%'
  AND station NOT LIKE '% AND %'
  AND station NOT LIKE '%CARHOUSE%'
  AND station NOT LIKE '%/%'
  AND station NOT LIKE '%UNKNOWN%'
  AND station NOT LIKE '%SUBWAY%'
  AND station NOT LIKE '%TRANSIT%'
  AND station NOT LIKE '%RAPID%'
  AND station NOT LIKE '% TO'
  AND station NOT LIKE '% TO %'
GROUP BY station
HAVING COUNT(*) >= 150;

CREATE INDEX idx_delays_date    ON delays (delay_date);
CREATE INDEX idx_delays_station ON delays (station);
CREATE INDEX idx_delays_code    ON delays (code);
