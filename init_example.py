
Shapelet_length = 4 # number of weeks ahead
Number_of_shapelets = 3 # number of shapelets to consider
shapelet_array = [[0]*Shapelet_length for w in range(Number_of_shapelets)]
shapelet_names = ["Inc", "Peak", "Flat","Dec"] 

shapelet_array[0] = [1,2,3,4] # inc
shapelet_array[1] = [1,2,2,1] # peak
shapelet_array[2] = [0,0,0,0] # flat
shapelet_array[3] = [4,3,2,1] # dec
