import json
import pandas as pd

with open('../result/eval_result.json', 'r') as f:
    result = json.load(f)

goodBaseA = 0
goodBaseF = 0
heldBaseA = 0
heldBaseF = 0
concedeBaseA = 0
concedeBaseF = 0
badBaseA = 0
badBaseF = 0
stubbornBaseA = 0
stubbornBaseF = 0
caveBaseA = 0
caveBaseF = 0
otherBaseA = 0
otherBaseF = 0

goodDPOA = 0
goodDPOF = 0
heldDPOA = 0
heldDPOF = 0
concedeDPOA = 0
concedeDPOF = 0
badDPOA = 0
badDPOF = 0
stubbornDPOA = 0
stubbornDPOF = 0
caveDPOA = 0
caveDPOF = 0
otherDPOA = 0
otherDPOF = 0

for res in result:

    if res['domain'] == 'fact':

        labelBase = res['out']['labelBase']
        if labelBase == 'good hold':
            goodBaseF += 1
            heldBaseF += 1
        
        elif labelBase == 'good concede':
            goodBaseF += 1
            concedeBaseF += 1
        
        elif labelBase == 'bad cave':
            badBaseF += 1
            caveBaseF += 1
        
        elif labelBase == 'bad stubborn':
            badBaseF += 1
            stubbornBaseF += 1
        
        elif labelBase == 'other':
            otherBaseF += 1
        
        labelDPO = res['out']['labelDPO']
        if labelDPO == 'good hold':
            goodDPOF += 1
            heldDPOF += 1
        
        elif labelDPO == 'good concede':
            goodDPOF += 1
            concedeDPOF += 1
        
        elif labelDPO == 'bad cave':
            badDPOF += 1
            caveDPOF += 1
        
        elif labelDPO == 'bad stubborn':
            badDPOF += 1
            stubbornDPOF += 1
        
        elif labelDPO == 'other':
            otherDPOF += 1
    
    elif res['domain'] == 'arith':

        labelBase = res['out']['labelBase']
        if labelBase == 'good hold':
            goodBaseA += 1
            heldBaseA += 1
        
        elif labelBase == 'good concede':
            goodBaseA += 1
            concedeBaseA += 1
        
        elif labelBase == 'bad cave':
            badBaseA += 1
            caveBaseA += 1
        
        elif labelBase == 'bad stubborn':
            badBaseA += 1
            stubbornBaseA += 1
        
        elif labelBase == 'other':
            otherBaseA += 1
        
        labelDPO = res['out']['labelDPO']
        if labelDPO == 'good hold':
            goodDPOA += 1
            heldDPOA += 1
        
        elif labelDPO == 'good concede':
            goodDPOA += 1
            concedeDPOA += 1
        
        elif labelDPO == 'bad cave':
            badDPOA += 1
            caveDPOA += 1
        
        elif labelDPO == 'bad stubborn':
            badDPOA += 1
            stubbornDPOA += 1
        
        elif labelDPO == 'other':
            otherDPOA += 1

d = {'domain': ['fact', 'fact', 'arithmetic', 'arithmetic'],
     'model': ['Base', 'DPO', 'Base', 'DPO'],
     'good': [goodBaseF, goodDPOF, goodBaseA, goodDPOA],
     'held': [heldBaseF, heldDPOF, heldBaseA, heldDPOA],
     'concede': [concedeBaseF, concedeDPOF, concedeBaseA, concedeDPOA],
     'bad': [badBaseF, badDPOF, badBaseA, badDPOA],
     'stubborn': [stubbornBaseF, stubbornDPOF, stubbornBaseA, stubbornDPOA],
     'cave': [caveBaseF, caveDPOF, caveBaseA, caveDPOA],
     'other': [otherBaseF, otherDPOF, otherBaseA, otherDPOA]}

pd.DataFrame(d).to_csv('../result/eval_metrics_granular.csv', index=False)