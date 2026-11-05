import sys, json
t = open(sys.argv[1]).read(); d = json.load(open(sys.argv[2]))
open(sys.argv[3], 'w').write(t.replace('/*GAMES*/null', json.dumps(d, separators=(',', ':'), ensure_ascii=False)))
