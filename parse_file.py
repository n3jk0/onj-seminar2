import json
import operator


def remove_duplicated_lines(filename):
    lines_seen = set()  # holds lines already seen
    outfile = open(filename + "_filtered", "w")
    comment_id = 0
    for line in open(filename, "r"):
        line_json = json.loads(line)
        body = str(line_json['body']).replace("\n", " ").replace("\r", "")
        if body not in lines_seen:  # not a duplicate
            new_line = "\t".join([str(comment_id), line_json['subreddit'], body])
            outfile.write(new_line + "\n")
            lines_seen.add(body)
            comment_id += 1
    outfile.close()
    print("remove_duplicated_lines DONE")
    print(comment_id)


def count_by_subreddit(filename):
    with open(filename, 'rt', encoding="utf8") as data_in:
        d = dict()
        for line in data_in:
            # print(line.split(";")[1])
            subreddit = line.split("\t")[1].lower()
            if subreddit in d:
                d[subreddit] += 1
            else:
                d[subreddit] = 1
    print("count_by_subreddit DONE")
    return sorted(d.items(), key=operator.itemgetter(1), reverse=True)


def filter_by_subreddit(filename, subreddits):
    with open(filename + "_final", "w") as outfile:
        with open(filename, 'rt', encoding="utf8") as data_in:
            dic = dict()
            for sub in subreddits:
                dic[sub] = 0
            for line in data_in:
                subreddit = line.split("\t")[1].lower()
                if subreddit in subreddits and dic[subreddit] < 1000:
                    dic[subreddit] += 1
                    outfile.write(line)
    print("filter_by_subreddit DONE")


FILE_NAME = "./data/RC_2018-01-01" # data with comments in onw day cc. 1.7 GB
FILTERED_FILE_NAME = FILE_NAME + "_filtered"
# remove_duplicated_lines(FILE_NAME)
# print(count_by_subreddit(FILTERED_FILE_NAME))
# According to count result We chose "politics" and "nba" subreddit
# In our new database we has 22079 comments from nfl and 22375 from nba subreddit
# [('politics', 22079), ('nba', 22375)]
filter_by_subreddit(FILTERED_FILE_NAME, ["politics", "nba"])
# File was renamed at the end to RC_2018-01-01_final
