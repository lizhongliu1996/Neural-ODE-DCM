data = read.csv("test_result_bonfer.csv")
data = data[,2:265]
data2 = data
diag(data) = 1

which(data != 1)
sum(data != 1)
data[data != 1]


matrix = read.csv("out/ncanda_result0.csv")
matrix = matrix[,2:265]

matrix2 = matrix

matrix[data == 1] = 0
matrix2[data2 == 1] = 0


write.csv(matrix, "beforr_fliter1.csv")
write.csv(matrix2, "beforr_fliter2.csv")
