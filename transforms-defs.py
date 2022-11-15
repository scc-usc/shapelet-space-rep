global Shapelet_length, Number_of_shapelets

### Helper function for computing our transformation 
#For shapelets that are not flat, we first compute correlation
#of the given shape with the shapelet. Operation is commutative
def similarity_non_flat(vector1,vector2):   
    if np.std(vector1) < 1e-100 or np.std(vector2) < 1e-100:
        similarity_value = 0
    else:
        similarity_value = pcor(vector1,vector2)[0][1]
    return similarity_value

### returns the shapelet name corresponding to the dimension with the 
# the highest value in the shapelet space representation
def return_best_shapelet(vector, slope_thres = 0.0005):
    corrs = shapelet_space_representation(vector, slope_thres)
    scenario = corrs.index(max(corrs))
    return shapelet_names[scenario]

### generate shapelet space representation for the given vector (Corr+slope version)
def shapelet_space_representation_gen(vector, slope_thres = 0.0005):

    coords = []
    # The average absolute slope of slope_thres 
    # gets a flatness of 0.1. Modify below to change
    beta = -log(0.1)/slope_thres
    
    # flat threshold is m0. If the slope is below m0 flatness is 1
    m0 = 0
    slope =mean(abs(diff(vector)))
    if slope < m0:
        flatness = 1
    else: 
        flatness = exp(-beta*(slope - m0));
    for i in range(len(shapelet_array)):
        if not(any(shapelet_array[i])):
            score = 2*flatness - 1
        else:
            score = (1-flatness)*similarity_non_flat(shapelet_array[i],vector)
        coords.append(score)
    return coords

# This representation approach only looks at the slope and decides the representation
def shapelet_space_representation_slope_based(vector, slope_thres = 0.0005):
  coords = []
  flatness = 0
  if abs(vector[-1] - vector[0]) < slope_thres:
        flatness = 1
  for i in range(len(shapelet_array)):
    score = 0
    if not(any(shapelet_array[i])):
      score = flatness
    else:
      if (vector[-1] - vector[0])*(shapelet_array[i][-1] - shapelet_array[i][0]) > 0:
        score = 1 - flatness
    coords.append(score)
  return coords


def shapelet_space_representation(vector, slope_thres = 0.0005, type = 'pearson'):
  if type == 'slope-based':
    return shapelet_space_representation_slope_based(vector, slope_thres)
  else:
    return shapelet_space_representation_gen(vector, slope_thres)


### Cosine similarity is used to measure performance, agreement
def cosine_sim(coords1, coords2):
    return 1 - spatial.distance.cosine(coords1,coords2)

### Takes a time-series and finds the shapelet space representations
# with a rolling window. The output is a multi-variate time-series
# of the same length encoding the shapes in the time-series. 
def find_shapelet_space_ts(time_series, slope_thres = 0.0005):
    reps = [[0.0]*(len(time_series)-Shapelet_length+1)]*len(shapelet_names)
    reps = np.array(reps)
    for i in range(0,len(time_series)-Shapelet_length+1):
        this_shape = time_series[i:i+Shapelet_length]
        this_rep = shapelet_space_representation(this_shape, slope_thres)
        reps[:, i] = this_rep
    return reps  
## Functions for generating shapelet representaiton for probabilistic forecasts provided with quantiles

def gen_sampled_shapes(quant_forecast, num_shapes):
  sampled_shapes = [[0]*len(quant_forecast)]*num_shapes
  sampled_shapes = np.array(sampled_shapes);
  for w in range(0,len(quant_forecast)):
    df = quant_forecast[w]
    df.dropna()
    min_q = df['quantile'].min()
    max_q = df['quantile'].max()
    f = interp1d(df['quantile'], df['value'], kind='linear')
    r_list = np.random.rand(num_shapes)
    #print(r_list)
    for n in range(0,num_shapes):
      r = min_q + (max_q - min_q)*r_list[n]
      sampled_shapes[n, w] = f(r)
  return sampled_shapes

def shapelet_space_representation_prob(quant_forecast, slope_thres = 0.0005, type = 'pearson', num_shapes = 100):
  shape_list = gen_sampled_shapes(quant_forecast, num_shapes)
  coords = [0]*Number_of_shapelets
  for s in range(0, num_shapes):
    vector = shape_list[s]
    if type == 'slope-based':
      these_coords = shapelet_space_representation_slope_based(vector, slope_thres)
    else:
      these_coords = shapelet_space_representation_gen(vector, slope_thres)
    
    for i in range(0,len(coords)):
      coords[i] = coords[i] + these_coords[i]
  for i in range(0,len(coords)):
    coords[i] = coords[i]/num_shapes
  return coords
