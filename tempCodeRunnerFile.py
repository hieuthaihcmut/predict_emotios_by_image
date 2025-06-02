        self.result = np.zeros((self.height_result,self.width_result))
        self.input = input
        self.max_or_avarage = max_or_avarage
    def operator(self):
        if(self.max_or_avarage==0):
            for row in range (0,self.height,2):
                for col in range (0,self.width,2):
                    self.result[row/2][col/2] = max(self.input[row][col],self.input[row+1][col],self.input[row][col+1],self.input[row+1][col+1])
            return self.result
        if(self.max_or_avarage==1):
            for row in range (0,self.height,2):
                for col in range (0,self.width,2):
                    self.result[row/2][col/2] = (self.input[row][col]+self.input[row+1][col]+self.input[row][col+1]+self.input[row+1][col+1])/4
            return self.result