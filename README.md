# SI1-2022-template

```python
if self.resources >= self.upgrade_cost: # upgrade building
    actions.append(upgradeBase())
    self.resources -= self.upgrade_cost 

# only buy ranged
default_cell_s_type = self.board[VCENTER][1][0]   # in numpy would be self.board[1,VCENTER,0]
if self.resources>=SOLDIER_RANGED_COST and default_cell_s_type in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
    buyamount = self.resources//SOLDIER_RANGED_COST
    actions.append( recruitSoldiers(ALLIED_SOLDIER_RANGED, self.resources//SOLDIER_RANGED_COST) )
    self.resources -= buyamount*SOLDIER_RANGED_COST


# example how to move troops from (4,0) to (4,1), step by step
origincell = self.board[0][4]   #in case of numpy array would be self.board[4,0]
targetcell = self.board[1][4]   #in case of numpy array would be self.board[4,1]
soldier_type, soldier_amount = targetcell
if soldier_type in [EMPTY_CELL, origincell[0]]:  # if target cell is empty or if contains same type troops 
    moveaction = moveSoldiers((4,0),(4,1),soldier_amount)
    actions.append(  moveaction )
```