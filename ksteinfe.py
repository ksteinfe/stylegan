"""Minimal script for generating an image using pre-trained StyleGAN generator."""
import os, time, pickle, progressbar
import cv2, imageio
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
#import config



def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # load model
    pth_pkl = r'X:\Box Sync\RRSYNC\temp\00001-sgan-190509_panogan_256-1gpu\network-snapshot-007726.pkl'
    pth_out = r'C:\Users\ksteinfe\Desktop\TEMP'

    generator, input_shape = load_saved_model(pth_pkl)

    # Pick latent vectors.
    for state in [0,1,2]:
        # state = 1 # change to pick other random vectors
        rnd = np.random.RandomState(state)
        img_count = 240

        '''
        # line
        pa, pb = rnd.randn(2, input_shape)
        latents = nd_line( pa, pb, img_count )
        #print("{}\t{}\t>\t{}".format(latents.shape, np.amin(latents),np.amax(latents)))
        '''

        for rad in [4,8,16,32,64]:
            pa, pb, origin = rnd.randn(3,input_shape)
            latents = nd_circle(input_shape, origin, pa, pb, rad, cnt=img_count)
            imgs = generate_images(latents, generator) # generate and save images
            animate_images(imgs,os.path.join(pth_out,"circle_{:02d}_{:02d}.mp4".format(state, rad)))

        #for n,img in enumerate(imgs): PIL.Image.fromarray(img,'RGB').save( os.path.join(pth_out, 'out_{}.png'.format(n)) )



def animate_images(src_imgs, pth_out, fps=30):
    print("creating animation...")
    secs_per_fade = 0.1000
    #secs_per_stil = 0.0 # NO STILL

    def fade(img1, img2, steps):
        ims = []
        for fad in np.linspace(0,1,steps):
            i1,i2 = np.asarray(img1), np.asarray(img2)
            arr = cv2.addWeighted( i1, 1-fad, i2, fad, 0)
            ims.append( arr )
        return ims

    imgs = []
    bar = progressbar.ProgressBar(max_value=len(src_imgs))
    for n, (img_a, img_b) in enumerate(zip(src_imgs[:-1],src_imgs[1:])):
        #for f in range(int(secs_per_stil/2.0*fps)): imgs.append(np.asarray(img_a))
        imgs.extend( fade(img_a, img_b, secs_per_fade * fps) )
        #for f in range(int(secs_per_stil/2.0*fps)): imgs.append(np.asarray(img_b))
        bar.update(n+1)
    bar.finish()

    start_time = time.process_time()
    print("writing mp4 with {} frames at {} fps to produce a {}s animation".format(len(imgs),fps,len(imgs)/fps))
    writer = imageio.get_writer(pth_out, fps=fps)
    for im in imgs: writer.append_data(im)
    writer.close()
    print("writing {}\ttook {:.2f}s".format(pth_out, time.process_time() - start_time))

def nd_line(pa,pb,cnt=48):
    return np.stack( [(pb-pa*t)+pa for t in np.linspace(0,1,cnt)], axis=0 )

def nd_circle(dims, origin, pa, pb, radius, cnt=48):
    #pa, pb, origin=np.random.rand(3,dims)
    u1=pa/np.linalg.norm(pa)
    u2=pb/np.linalg.norm(pb)
    V3=u1-np.dot(u1,u2)*u2
    u3=V3/np.linalg.norm(V3)
    #print(np.linalg.norm(u2),np.linalg.norm(u3),np.dot(u2,u3))
    #1.0 1.0 0.0
    #u2,u3 is orthonormed
    theta=np.arange(0,2*np.pi,2*np.pi/cnt)

    pts = origin+radius*(np.outer(np.cos(theta),u2)+np.outer(np.sin(theta),u3))
    return np.stack(pts)


def load_saved_model(pth_pkl):
    # Load pre-trained network
    with open( pth_pkl, "rb" ) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    #Gs.print_layers()
    return Gs, Gs.input_shape[1]

def generate_images(latents, generator):
    print("generating images...")
    start_time = time.process_time()
    print("Given generator expects an input of shape {2}.\nGenerating images from a collection of {0} latent vectors each of which has a shape of {1}".format(latents.shape[0],latents.shape[1],generator.input_shape[1]))

    # Generate images.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = False
    batch_size = 12
    if latents.shape[0] < batch_size:
        images = generator.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    else:
        images = []
        batches = [latents[i:i + batch_size] for i in range(0, len(latents), batch_size)]
        print("Splitting into {} batches of {} images each.".format(len(batches), batch_size))
        bar = progressbar.ProgressBar(max_value=len(batches))
        for b, batch_latents in enumerate(batches):
            images.extend( generator.run(batch_latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt) )
            bar.update(b+1)
        bar.finish()

    print("Generated {} images in {}s".format(len(images), time.process_time()-start_time))
    return(images)

if __name__ == "__main__":
    main()
