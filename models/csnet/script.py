import pandas as pd
from data import teams_dict, maps, BuyType

train = pd.read_excel('395.xlsx', engine='openpyxl')
print(len(train['tTeam'].unique()))

for i in range(len(train['ctTeam'])):
    print(i)
    train.at[i, 'mapName'] = maps[train.loc[i]['mapName']]
    train.at[i, 'ctBuyType'] = BuyType[train.loc[i]['ctBuyType']]
    train.at[i, 'tBuyType'] = BuyType[train.loc[i]['tBuyType']]

    if train.loc[i]['losingTeam'] == train.loc[i]['ctTeam']:
        train.at[i, 'ctResult'] = 0
        train.at[i, 'tResult'] = 1
    else:
        train.at[i, 'ctResult'] = 1
        train.at[i, 'tResult'] = 0

    train.at[i, 'losingTeam'] = teams_dict[train.loc[i]['losingTeam']]
    train.at[i, 'winningTeam'] = teams_dict[train.loc[i]['winningTeam']]

    train.at[i, 'ctTeam'] = teams_dict[train.loc[i]['ctTeam']]
    train.at[i, 'tTeam'] = teams_dict[train.loc[i]['tTeam']]

    train.at[i, 'ctFreezeTimeEndEqVal'] = train.loc[i]['ctFreezeTimeEndEqVal'] / 1000
    train.at[i, 'tFreezeTimeEndEqVal'] = train.loc[i]['tFreezeTimeEndEqVal'] / 1000
    train.at[i, 'ctRoundStartEqVal'] = train.loc[i]['ctRoundStartEqVal'] / 1000
    train.at[i, 'tRoundStartEqVal'] = train.loc[i]['tRoundStartEqVal'] / 1000

train.to_excel("output1.xlsx", index=False)

