close all
clear

cards = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "Ace", "Jack", "King", "Queen"];
acts = imread('../dense_acts.png');

sneeze = tsne(double(acts));
classes = uint32(floor((0:779)/60));
class_names=cards(classes+1);

gscatter(sneeze(:,1),sneeze(:,2),class_names)
