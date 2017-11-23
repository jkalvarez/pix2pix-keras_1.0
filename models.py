from keras.models import Model

from keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate

from keras.initializers import RandomNormal

def unet_conv(x, n_filters, name, bn=True):
    # Convolution
    x = Conv2D(n_filters, kernel_size=4, strides=2, name=name, padding='same',
        use_bias=True, kernel_initializer='random_normal')(x)
    
    # Batch normalization
    if bn:
        x = BatchNormalization(momentum=0.9, epsilon=1.01e-5,
        gamma_initializer=RandomNormal(1., 0.02))(x)
    
    # ReLu activation
    x = LeakyReLu(0.2)(x)

    return x
    
    
def unet_deconv(x, x2, n_filters, name, bn=True, dropout=False):
    # Deconvolution
    x = Conv2DTranspose(n_filters, kernel_size=4, strides=2, name=name, 
        use_bias=False, kernel_initializer = RandomNormal(1., 0.02))(x)        
    x = Cropping2D(1)(x)    

    # Batch Normalization
    if bn:
        x = BatchNormalization(momentum=0.9, epsilon=1.01e-5,
        gamma_initializer=RandomNormal(1., 0.02))(x)
    
    # Dropout
    if dropout:
         x = Dropout(0.5)(x)

    # ReLu activation
    x = LeakyReLu(0.2)(x)
    
    # unet skip concatenation
    x = Concatenate(axis=-1)([x, x2])

    return x

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization
channel_axis = -1 #tensorflow

def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)

def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
    gamma_initializer = gamma_init)

def unet(im_size, nc_in=1, nc_out=1, ngf=64, fixed_input_size=True):
    
    # Encoder
    # C64-C128-C256-C512-C512-C512-C512-C512. Hell of a bottleneck for a 256 image 
    
    # Decoder
    # CD512-CD512-CD512-C512-C512-C256-C128-C64

    
    max_nf = 8*ngf    
    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = LeakyReLU(alpha=0)(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    s = im_size if fixed_input_size else None
    _ = inputs = Input(shape=(s, s, nc_in))        
    _ = block(_, im_size, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])

# Discriminator
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))
    _ = Concatenate(axis=channel_axis)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid") (_)    
    return Model(inputs=[input_a, input_b], outputs=_)
        
