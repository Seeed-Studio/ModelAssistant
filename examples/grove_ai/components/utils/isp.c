/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      isp.h
* @brief     
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-19
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#include <stdio.h>
#include <stdint.h>
#include "isp.h"
#include "grove_ai_config.h"
#include "logger.h"

#define CLIP(value) (unsigned char)(((value) > 0xFF) ? 0xff : (((value) < 0) ? 0 : (value)))

void yuv422p2rgb(uint8_t *pdst, const uint8_t *psrc, int h, int w, int c, int target_h, int target_w,uint8_t rotation)
{
  int32_t y;
  int32_t cr;
  int32_t cb;

  int32_t r, g, b;
  uint32_t init_index, cbcr_index, index;
  uint32_t pixs = w * h;
  uint32_t u_chunk = w * h;
  uint32_t v_chunk = w * h + w * h / 2;
  float beta_h = (float)h / target_h, beta_w = (float)w / target_w;

  for (int i=0; i< target_h; i++)
  {
    for (int j=0; j<target_w; j++)
    {
      int tmph = i * beta_h, tmpw = beta_w * j;
      // select pixel
      index = i * target_w + j;
      init_index = tmph * w + tmpw;    //ou
      cbcr_index = init_index % 2 ? init_index - 1 : init_index;

      y = psrc[init_index];
      cb = psrc[u_chunk + cbcr_index / 2];
      cr = psrc[v_chunk + cbcr_index / 2];
      r = (int32_t)(y + (14065 * (cr - 128)) / 10000);
      g = (int32_t)(y - (3455 * (cb - 128)) / 10000 - (7169 * (cr - 128)) / 10000);
      b = (int32_t)(y + (17790 * (cb - 128)) / 10000);

      switch (rotation)
      {
      case ROTATION_LEFT:
        index = (target_w - 1 - index % target_w) * (target_h) + index / target_w;
        break;
      case ROTATION_RIGHT:
        index = (index % target_w) * (target_h) + (target_h - 1 - index / target_w);
        break;
      default:
        break;
      }
      if (c == 1)
      {
        // rgb to gray
        uint8_t gray = (r * 299 + g * 587 + b * 114) / 1000;
        pdst[index] = (uint8_t)CLIP(gray);
      }
      else if (c == 3)
      {
        pdst[index * 3 + 0] = (uint8_t)CLIP(r);
        pdst[index * 3 + 1] = (uint8_t)CLIP(g);
        pdst[index * 3 + 2] = (uint8_t)CLIP(b);
      }
    } 
  }
}