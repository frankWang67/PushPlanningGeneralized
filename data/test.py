import json

test_data = {'node': 'root', 
             'id': 24524928451,
             'state': [0, 0, 0],
             'path': [[0,0,0],[1,1,1],[2,2,2]],
             'scene': 'http://127.0.0.1:8787/Figure_1.png',
             'parent': 'null',
             'child': [{'node': 'node_0',
                        'id': 34896258953,
                        'scene': 'http://127.0.0.1:8787/Figure_2.png'},
                       {'node': 'node_1',
                        'id': 10471572195,
                        'scene': 'http://127.0.0.1:8787/Figure_3.png'}]}

with open("./test.json", "w") as f:
    f.write(json.dumps(test_data, ensure_ascii=False, indent=4, separators=(',', ':')))
