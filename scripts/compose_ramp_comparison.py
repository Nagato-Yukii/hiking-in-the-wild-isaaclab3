"""Compose first-episode checkpoint playback; freeze completed episode before autoreset."""
import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];art=root/'artifacts';font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
folders=[root/'logs/video-ramp-15000-20261105',root/'logs/video-ramp-17000-20261105']
filters=[];inputs=[];metadata=[]
for i,(folder,iteration) in enumerate(zip(folders,[15000,17000])):
 row=next(r for r in json.loads((folder/'local_episodes.json').read_text()) if r['terrain']=='ramp')
 metadata.append(row);inputs += ['-i',str(folder/'videos/clip_0000.mp4')]
 duration=max(.02,min(15.,row['seconds']-.04 if row['status']==1 else 15.))
 filters.append(f'[{i}:v]trim=duration={duration},setpts=PTS-STARTPTS,scale=640:480,tpad=stop_mode=clone:stop_duration=15,trim=duration=15[v{i}]')
 status=f"{row['seconds']:.2f}秒到达；完成前末帧定格" if row['status']==1 else '15秒超时，未到达目标'
 (art/f'ramp_label_{i}.txt').write_text(f'{iteration} 更新 | '+status)
(art/'ramp_footer.txt').write_text('同一初始状态、同一斜坡；左：续训前，右：续训后\n单次回放示例；1000次评估成功率：93.2% → 0%')
filters.append('[v0][v1]hstack=inputs=2,pad=1280:608:0:64:color=0x151c28[stack]')
filters.append(f"[stack]drawtext=expansion=none:fontfile={font}:textfile={art}/ramp_label_0.txt:fontsize=21:fontcolor=white:x=14:y=21,drawtext=expansion=none:fontfile={font}:textfile={art}/ramp_label_1.txt:fontsize=21:fontcolor=white:x=654:y=21,drawtext=expansion=none:fontfile={font}:textfile={art}/ramp_footer.txt:fontsize=20:fontcolor=white:x=14:y=552:line_spacing=5[out]")
output=art/'G1_斜坡_15000_vs_17000_续训前后.mp4'
subprocess.run(['ffmpeg','-y','-loglevel','error',*inputs,'-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-t','15',str(output)],check=True)
assert json.loads((folders[0]/'initial_state_signatures.json').read_text())==json.loads((folders[1]/'initial_state_signatures.json').read_text())
(art/'ramp_video_provenance.json').write_text(json.dumps({'episodes':metadata,'initial_states_match':True,'output':str(output),'freeze_success_before_autoreset':True},indent=2));print(output)
