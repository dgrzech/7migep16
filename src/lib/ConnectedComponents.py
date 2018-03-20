import numpy as np


class ConnectedComponents2D(object):
    def __init__(self, binary):
        self.labels = []
        self.binary_labelling = np.zeros(np.shape(binary))

        label_numbers_for_unification = []

        dims =np.shape(binary)

        current_label = self.pick_label()

        # Time to search

        for i in np.arange(dims[0], dtype=int):
          print i
          for j in np.arange(dims[1], dtype=int):
              #print(i, j, k)
              if binary[i][j] == 1:
                # Normal Case
                #cells_to_check = [binary[i-1][j][k], binary[i-1][j-1][k], binary[i][j-1][k], binary[i][j][k-1], binary[i-1][j][k-1], binary[i-1][j-1][k-1], binary[i][j-1][k-1] ]
                cells_to_check = [ [i-1,  j], [i-1, j-1], [i, j-1], [i+1, j-1] ] # 8 connectivity
                cells_to_check = [ [i-1,  j], [i-1, j-1], [i-1, j+1], [i, j-1] ] # 8 connectivity


                labelled = False
                labelled_with = []

                if((i==0) & (j==0)):
                  pass
                else:
                  for cell in cells_to_check:
                    # Check that the cells do not exceed the image boundaries (8-connectivity)
                    if((cell[0]<0) |(cell[1]<0) |(cell[0]>dims[0]) |(cell[1]>dims[1]) ):
                        pass # cell is outside domain
                    else:
                        this_cell_label = self.binary_labelling[cell[0]][cell[1]]
                        if (this_cell_label > 0):
                          #print(this_cell_label, self.binary_labelling[i][j][k], this_cell_label<self.binary_labelling[i][j][k], labelled==True )
                          if ( (labelled==False) | ( (labelled==True) & (this_cell_label<self.binary_labelling[i][j])) ) :
                            self.binary_labelling[i][j] = this_cell_label
                          labelled = True
                          labelled_with.append(this_cell_label)
                        if(len(labelled_with)>1):
                          label_numbers_for_unification.append(labelled_with)

                if labelled==False:
                  self.binary_labelling[i][j] = self.pick_label()

        old_label_list = np.copy(self.labels)
        # Labels all found. Now it is time to union
        for ii in np.arange(len(label_numbers_for_unification)):
          i = (len(label_numbers_for_unification)-1)-ii
          this_label_set = label_numbers_for_unification[i]
          smallest_label = np.min(this_label_set)
          for this_label in this_label_set:
            if this_label != smallest_label:
              self.binary_labelling[self.binary_labelling==this_label]=smallest_label
              del(self.labels[self.labels.index(this_label)])
              #############################################
              # Why are the below lines of code neccessary?
              for jj in np.arange(len(label_numbers_for_unification)):
                for jj_thislabel_no in np.arange(len(label_numbers_for_unification[jj])):
                  if(label_numbers_for_unification[jj][jj_thislabel_no]==this_label):
                    label_numbers_for_unification[jj][jj_thislabel_no]=smallest_label



class ConnectedComponents3D(object):

  def __init__(self, binary):
    self.labels = []
    self.binary_labelling = np.zeros(np.shape(binary))

    label_numbers_for_unification = []

    dims =np.shape(binary)

    current_label = self.pick_label()

    # Time to search

    for i in np.arange(dims[0], dtype=int):
      print i
      for j in np.arange(dims[1], dtype=int):
        for k in np.arange(dims[2], dtype=int):
          #print(i, j, k)
          if binary[i][j][k] == 1:
            # Normal Case
            '''
            cells_to_check = [ [i-1,  j, k], [i-1, j-1, k], [i,j-1,k], [i,j,k-1], [i-1,j,k-1], [i-1,j-1,k-1], [i,j-1,k-1] ]

            if((i==0 )    &   ( j!=0 )    &   ( k!=0)):
              cells_to_check = [ [i,j-1,k], [i,j,k-1], [i,j-1,k-1] ]
            if((i!=0 )    &   ( j==0 )    &   ( k!=0)):
              cells_to_check = [ [i-1,  j, k],  [i,j-1,k], [i,j,k-1], [i-1,j,k-1], ]
            if((i!=0 )    &   ( j!=0 )    &   ( k==0)):
              cells_to_check = [ [i-1,  j, k], [i-1, j-1, k], [i,j-1,k] ]
            if((i==0 )    &   ( j==0 )    &   ( k!=0)):
              cells_to_check = [ [i,j,k-1]]
            if((i!=0 )    &   ( j==0 )    &   ( k==0)):
              cells_to_check = [ [i-1,  j, k] ]
            if((i==0 )    &   ( j!=0 )    &   ( k==0)):
              cells_to_check = [ [i,j-1,k] ]
            '''
            cells_to_check = [ [i, j, k-1], [i, j-1, k-1], [i, j-1, k], [i, j-1, k+1], # layer 1 connectivity
            [i-1, j-1, k-1],[i-1, j-1, k], [i-1, j-1, k+1], [i-1, j, k-1],[i-1, j, k],
            [i-1, j, k+1], [i-1, j+1, k-1],[i-1, j+1, k], [i-1, j+1, k+1]] # layer 2 connectivity

            labelled = False
            labelled_with = []

            if((i==0) & (j==0) & (k==0)):
              pass
            else:
              for cell in cells_to_check:
                # Check that the cells do not exceed the image boundaries (8-connectivity)
                if((cell[0]<0) |(cell[1]<0) |(cell[2]<0) |(cell[0]>dims[0]) |(cell[1]>dims[1]) |(cell[2]>dims[2])  ):
                    pass # cell is outside domain
                else:
                    this_cell_label = self.binary_labelling[cell[0]][cell[1]][cell[2]]
                    if (this_cell_label > 0):
                      #print(this_cell_label, self.binary_labelling[i][j][k], this_cell_label<self.binary_labelling[i][j][k], labelled==True )
                      if ( (labelled==False) | ( (labelled==True) & (this_cell_label<self.binary_labelling[i][j][k])) ) :
                        self.binary_labelling[i][j][k] = this_cell_label
                      labelled = True
                      labelled_with.append(this_cell_label)
                    if(len(labelled_with)>1):
                      label_numbers_for_unification.append(labelled_with)

            if labelled==False:
              self.binary_labelling[i][j][k] = self.pick_label()

    old_label_list = np.copy(self.labels)
    # Labels all found. Now it is time to union
    for ii in np.arange(len(label_numbers_for_unification)):
      i = (len(label_numbers_for_unification)-1)-ii
      this_label_set = label_numbers_for_unification[i]
      smallest_label = np.min(this_label_set)
      for this_label in this_label_set:
        if this_label != smallest_label:
          self.binary_labelling[self.binary_labelling==this_label]=smallest_label
          del(self.labels[self.labels.index(this_label)])
          #############################################
          # Why are the below lines of code neccessary?
          for jj in np.arange(len(label_numbers_for_unification)):
            for jj_thislabel_no in np.arange(len(label_numbers_for_unification[jj])):
              if(label_numbers_for_unification[jj][jj_thislabel_no]==this_label):
                label_numbers_for_unification[jj][jj_thislabel_no]=smallest_label


  def pick_label(self):
    i=1
    no_found = False
    while no_found == False:
      if i in self.labels:
        i +=1
      else:
        no_found=True
    self.labels.append(i)
    return i

  def n_largest_components(self, n=2):
    num_components_to_keep = n

    biggest_val = 0.
    biggest_component_label = 0.

    values = []
    component_labels = []



    for label in self.labels:
      mask = (self.binary_labelling==label)
      val = np.sum(mask)
      values.append(val)
      component_labels.append(label)

    sorted_values, sorted_component_labels = zip(*[(x, y) for x, y in sorted(zip(values, component_labels))])
    print(sorted_values)
    print(sorted_component_labels)

    connected_component = np.zeros(np.shape(self.binary_labelling))
    num_components = len(sorted_values)
    for i in np.arange(num_components_to_keep):
      this_label = sorted_component_labels[(num_components-1) - i]
      connected_component = (  (connected_component==True) | (self.binary_labelling==this_label) )

    return connected_component
  def n_largest_components_old(self):
    biggest_val = 0.
    biggest_component_label = 0.



    for label in self.labels:
      mask = (self.binary_labelling==label)
      val = np.sum(mask)
      if val>biggest_val:
        biggest_val = np.copy(val)
        biggest_component_label = np.copy(label)


    print(biggest_component_label)
    print(biggest_val)
    connected_component = (self.binary_labelling==biggest_component_label)
    return connected_component
