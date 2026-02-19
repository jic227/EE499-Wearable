from stats_functions import *

# Simple test dataset
data = [10, 20, 30, 40, 50]

print("Arithmetic Mean:", arithmetic_mean(data))
print("Harmonic Mean:", harmonic_mean(data))
print("Standard Deviation:", standard_deviation(data))

std1 = 10
std2 = 12
n1 = 30
n2 = 35

print("Pooled Std:", pooled_std([std1, std2], [n1, n2]))

group1 = [10, 12, 14, 16, 18]
group2 = [20, 22, 24, 26, 28]

t, p = t_test(data1=group1, data2=group2)

print("T-value:", t)
print("P-value:", p)

g1 = [10, 12, 14]
g2 = [20, 22, 24]
g3 = [30, 32, 34]

F, p = one_way_anova(g1, g2, g3)

print("F-value:", F)
print("ANOVA p-value:", p)

rm_data = [
    [10, 12, 14],
    [11, 15, 13],
    [9, 11, 16]
]

F_rm, p_rm = repeated_measures_anova(rm_data)

print("Repeated Measures F:", F_rm)
print("Repeated Measures p:", p_rm)