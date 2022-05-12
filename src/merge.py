import sys

shift = int(sys.argv[1])
top = int(sys.argv[2])
total_part = int(sys.argv[3])

f_list = []
for part in range(total_part):
    f0 = open('output/res.top%s-part%s' % (top, part))
    f_list.append(f0)

line_list = []
for part in range(total_part):
    line = f_list[part].readline()  # 一个文件读一行
    line_list.append(line)  # \分割：qid, pid, rank, score

out = open('output/dev.res.top%s' % top, 'w')
last_q = ''  # 上一个query
ans_list = {}
while line_list[-1]:
    cur_list = []
    for line in line_list:  # 把当前每个文件读的split了
        sub = line.strip().split('\t')
        cur_list.append(sub)

    if last_q == '':  # 之前没有query，就存第一个
        last_q = cur_list[0][0]  # qid
    if cur_list[0][0] != last_q:  # 当前cur_list的第一个，不是上一个query了（证明一个query查找结束）
        rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)  # 排序
        for i in range(top):
            out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
        ans_list = {}  # 清空当前query的pid list
    for i, sub in enumerate(cur_list):  # 遍历cur_list, 加进去
        ans_list[int(sub[1]) + shift*i] = float(sub[-1])  # ans_list[pid]
    last_q = cur_list[0][0]

    line_list = []  # 一次性只存4个处理，一个文件一个，肯定对应的都是相同的qid、rank
    for f0 in f_list:
        line = f0.readline()
        line_list.append(line)

rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
for i in range(top):
    out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
out.close()

print('output/dev.res.top%s' % top)
