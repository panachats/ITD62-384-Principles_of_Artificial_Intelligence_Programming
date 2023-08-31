import pandas as pd
import cv2
import matplotlib.pyplot as plt
# read the image
img = cv2.imread("D:\\walailak\\Year_2_Semester_3\pythonProject\\cat.png")
# resize image
output = cv2.resize(img, (9, 9))
frameGRAY = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
plt.imshow(frameGRAY)
plt.title(" frameGRAY ")
plt.show()


# Transform 2D array into Dataframe df2
df = pd.DataFrame(frameGRAY)
# Stack the prescribed level(s) from columns to index.
# Return a reshaped DataFrame
df = pd.DataFrame((df.stack(0)))
df.reset_index(inplace=True)
df2 = df.rename(columns={'level_0':'Y','level_1':'X',0:'GRAY'})

print("frameGRAY: ", frameGRAY.shape)
print(frameGRAY)
print(df2)
