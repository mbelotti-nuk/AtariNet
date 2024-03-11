import imageio

gif_original = 'match.gif'
gif_speed_up = 'foo.gif'

gif = imageio.mimread(gif_original, memtest=False)

imageio.mimsave(gif_speed_up, gif, duration=20)