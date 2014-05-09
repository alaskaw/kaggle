DROP TABLE IF EXISTS RAW_TRAIN;
CREATE TABLE RAW_TRAIN(
ID BIGINT PRIMARY KEY
,TITLE TEXT
,BODY TEXT
,TAGS TEXT
);
copy RAW_TRAIN from '/Users/thomasbuhrmann/Code/kaggle/facebook/data/Train.csv' csv header;

DROP TABLE IF EXISTS RAW_TEST; 
CREATE TABLE RAW_TEST(
ID BIGINT PRIMARY KEY
,TITLE TEXT
,BODY TEXT
);
copy RAW_TEST from '/Users/thomasbuhrmann/Code/kaggle/facebook/data/Test.csv' csv header;
