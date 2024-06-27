In the project directory, you can run:

### `npm start`

### How it Works ###
  - There are two buttons, Input a Player and Most Overvalued/Undervalued Players
  - Input a Player:
      - Takes player value inputs, which goes to a backend FastAPI server containing a ML model that predicts the valuations of football players between 2017 and 2021 (Model R^2: 0.87). 
      - Then, it outputs the player's predicted valuation, and compares it against their actual valuation, inputting whether they are overvalued or undervalued
  - Most Overvalued/Undervalued Players:
      - Based on my model, it predicts the valuations of football players and outputs the deficit between their valuation and predicted valuation, outputting the top 25 more overvalued and undervalued players based on my model. 


