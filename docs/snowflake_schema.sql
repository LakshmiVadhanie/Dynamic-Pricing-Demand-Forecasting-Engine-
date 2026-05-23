-- docs/snowflake_schema.sql
-- Favorita dataset schema for Snowflake
-- Load CSVs from Kaggle into these tables before running the pipeline

CREATE DATABASE IF NOT EXISTS FAVORITA;
USE DATABASE FAVORITA;
CREATE SCHEMA IF NOT EXISTS RAW;
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS TRAIN (
    id          BIGINT,
    date        DATE        NOT NULL,
    store_nbr   INTEGER     NOT NULL,
    item_nbr    INTEGER     NOT NULL,
    unit_sales  FLOAT       NOT NULL,
    onpromotion BOOLEAN
);

CREATE TABLE IF NOT EXISTS ITEMS (
    item_nbr    INTEGER     PRIMARY KEY,
    family      VARCHAR(50) NOT NULL,
    class       INTEGER,
    perishable  BOOLEAN
);

CREATE TABLE IF NOT EXISTS STORES (
    store_nbr   INTEGER     PRIMARY KEY,
    city        VARCHAR(50),
    state       VARCHAR(50),
    type        CHAR(1),
    cluster     INTEGER
);

CREATE TABLE IF NOT EXISTS OIL (
    date        DATE    PRIMARY KEY,
    dcoilwtico  FLOAT
);

CREATE TABLE IF NOT EXISTS HOLIDAYS_EVENTS (
    date        DATE,
    type        VARCHAR(20),
    locale      VARCHAR(10),
    locale_name VARCHAR(50),
    description VARCHAR(200),
    transferred BOOLEAN
);

-- Recommended clustering keys for query performance
ALTER TABLE TRAIN CLUSTER BY (date, store_nbr);
ALTER TABLE ITEMS CLUSTER BY (family);
