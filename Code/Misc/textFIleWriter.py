
text_file = open("../TxtFiles/trainName.txt", "w")
for i in range(1, 20401):
    text_file.write("TrainImages"+"/%#08d\n" % (i))
text_file.close()
