import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, temperature, alpha):
        super(KnowledgeDistillationLoss, self).__init__()
        self.get_cross_entropy_loss = nn.CrossEntropyLoss()
        self.get_KL_divergence_loss = nn.KLDivLoss(reduction='batchmean')
        self.temperature = temperature
        self.alpha = alpha
        # self.teacher_model = teacher_model
    
    def forward(self, teacher_outputs, student_logits, labels):
        # teacher_outputs = self.teacher_model(input_images)
        cross_entropy_loss = self.get_cross_entropy_loss(student_logits, labels)
        kl_divergence_loss = self.get_KL_divergence_loss(F.log_softmax(student_logits/self.temperature, dim=1), F.softmax(teacher_outputs/self.temperature, dim=1))
        total_loss = cross_entropy_loss*self.alpha + kl_divergence_loss*(1 - self.alpha)*self.temperature*self.temperature
        # import pdb;pdb.set_trace()
        return total_loss