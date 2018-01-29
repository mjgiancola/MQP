class Label():
  def __init__(self, imageIdx, labelerId, label):
    self.imageIdx = imageIdx   # Unique identifier for a particular image
    self.labelerId = labelerId # Unique identifier for a particular labeler
    self.label = label         # Response from labelerId for imageIdx

  def __str__(self):
  	return "%d %d %d\n" % (self.imageIdx, self.labelerId, self.label)