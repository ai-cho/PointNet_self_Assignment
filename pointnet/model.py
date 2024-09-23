import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)  
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), # input channel, output channel, kernel size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3] N: # of points
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.

        pointcloud = pointcloud.transpose(1,2) # (B, 3, N)
        if self.input_transform:
            it_output = self.stn3(pointcloud)
            pointcloud = torch.matmul(it_output, pointcloud) # (B, 3 , N) 3이 channel 수

        pointcloud = self.mlp1(pointcloud) # (B, 64, N)

        if self.feature_transform:
            ft_output = self.stn64(pointcloud)
            pointcloud = torch.matmul(ft_output, pointcloud) # (B, 64, N)

        pointcloud = self.mlp2(pointcloud) # (B, 1024, N)
        pointcloud = torch.max(pointcloud, 2)[0] # (B, 1024)

        return pointcloud

class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform) # (B, 1024)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes) # 출력층에서는 batchnorm이나 relu를 사용하지 않음. 최종 예측값을 왜곡시킬 수 있음.
            )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3] 
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.

        output_logit = self.pointnet_feat(pointcloud) # (B, 1024)
        output_logit = self.mlp(output_logit) # (B, num_classes)
        
        return output_logit


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.stn3 = STNKd(k=3)  
        self.stn64 = STNKd(k=64)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), # input channel, output channel, kernel size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, m, 1)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        pointcloud = pointcloud.transpose(1,2) # (B, 3, N)
        if self.input_transform:
            it_output = self.stn3(pointcloud)
            pointcloud = torch.matmul(it_output, pointcloud) # (B, 3 , N) 3이 channel 수

        feature_transform = self.mlp1(pointcloud) # (B, 64, N)
        pointcloud = feature_transform.copy()

        if self.feature_transform:
            ft_output = self.stn64(pointcloud)
            pointcloud = torch.matmul(ft_output, pointcloud) # (B, 64, N)

        pointcloud = self.mlp2(pointcloud) # (B, 1024, N)
        pointcloud = torch.max(pointcloud, 2)[0] # (B, 1024)

        global_feature = pointcloud.copy()

        global_feature = global_feature.unsqueeze(2).repeat(1, 1, feature_transform.shape[2])
        
        changed_input = torch.cat((feature_transform, global_feature), 1) # (B, 1088, N)
        changed_input= self.mlp3(changed_input) # (B, 128, N)
        output = self.mlp4(changed_input) # (B, m=50, N)
        
        return output


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points
        self.mlp1 = nn.Sequential(
            nn.Linear(1024, self.num_points/4),
            nn.BatchNorm1d(self.num_points/4),
            nn.ReLU(),
            nn.Linear(self.num_points/4, self.num_points/2),
            nn.BatchNorm1d(self.num_points/2),
            nn.ReLU(),
            nn.Linear(self.num_points/2, self.num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(self.num_points, self.num_points*3)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.

        pointcloud = self.pointnet_feat(pointcloud) # [B, 1024]
        pointcloud = self.mlp1(pointcloud) # [B, num_points*3]
        pointcloud = pointcloud.reshape(-1, self.num_points, 3)

        return pointcloud
        


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
