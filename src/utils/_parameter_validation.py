import numpy as np
#%%
class Interval:
    def __init__(self,data_type,left,right,closed):
        """closed_dict
         left,right,both,neither"""
        self.data_type = data_type
        self.left = left
        self.right = right
        self.closed = closed
        


    def validate(self,value):
        _RULES = {
            "left": lambda x,l,r: l <= x and x < r,
            "right": lambda x,l,r: l < x and x < r,
            "neither" : lambda x,l,r: l <= x and x <= r,
            "both": lambda x,l,r: l < x and x < r
        
        }
        rule_function = _RULES[self.closed]#atribuindo uma funcao a uma variavel
        return AssertionError( rule_function(value,self.left,self.right) and isinstance(value, int))  # noqa: E501

## comparar memoria com estado
## var_1 is var_2 (mesmo id ?)
## var_1 ==var_2 (mesmo estado?)
# %%
p = Interval(int,1,3,"both")
p.validate(1)
# %%
a =4
isinstance(a,int)
# %%
help(int)
# %%
def base_n(value,n):
    digits = []
    while value>0:
        m=value%n
        value = value//n
        digits.insert(0,m)
    return digits
base_n(55,3)
# %%
