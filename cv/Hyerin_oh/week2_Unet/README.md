## UNET : https://arxiv.org/abs/1505.04597

### 요약 정리
- Contracting path와 Expanding path가 대칭을 이루는 형태
    - Contracting path : context를 잡아냄
        - 3 x 3 conv를 padding 없이 사용해 h,w가 2씩 줄어드는 것이 특징
        - 2 x 2 max pooling 진행 시 , 정보 보존을 위해 channel 수를 2배 늘림
    - Expading path : 정확한 precision을 위함
        - Contracting path와 동일하게 진행 , max pooling을 up-conv로 대체
        - 깊이가 동일한 Contracting path의 layer에서 feature map을 받아온다. (skip connection)
        - 채널이 줄어들면서 잃어버린 고해상의 정보들을 보충해주는 작업
        - skip connection이 진행될 때는 , H,W사이즈가 다르므로 center crop을 통해 크기를 맞춰줌

- Elastic deformation 사용
    - 일종의 왜곡을 주는 효과로써, 세포같이 형체가 불일정하고 꾸불꾸불한 곳에 효과적

- weighted loss 사용
    - cell 들을 잘 구분하기 위해서 boundary를 잘 잡아내야함 → 가장 가까운, 2번째로 가까운 셀과의 거리를 이용하여 weighted loss로 해결

### 코드 정리
- 문제점 : input size가 572가 아닐 때는 작동하지 않는다. img의 size를 list로 입력해주는 별도의 작업이 필요할 거 같다.
- py로 작성 후 colab으로 돌려보려고 했지만, input과 output의 size가 달라 loss계산이 되지 않아 학습은 하지 못했다.

![image](https://user-images.githubusercontent.com/68813518/117663015-2a443780-b1db-11eb-877c-4730986c93a0.png)
