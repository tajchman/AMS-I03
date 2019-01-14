/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include "io_png.hxx"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

void abort_(const char * s, ...)
{
  va_list args;
  va_start(args, s);
  vfprintf(stderr, s, args);
  fprintf(stderr, "\n");
  va_end(args);
  abort();
}

cImage read_png_file (char *file_name)
{
  int nb, x, y;
  
  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep * row_pointers;
  png_const_byte header[8];    // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(file_name, "rb");
  if (!fp)
    abort_("[read_png_file] File %s could not be opened for reading", file_name);
  fread(header, 1, 8, fp);
  if (png_sig_cmp(header, 0, 8))
    abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    abort_("[read_png_file] png_create_read_struct failed");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    abort_("[read_png_file] png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[read_png_file] Error during init_io");

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  cImage I(png_get_image_width(png_ptr, info_ptr),
	   png_get_image_height(png_ptr, info_ptr));
  I.color_type = png_get_color_type(png_ptr, info_ptr);
  I.bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);


  /* read file */
  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[read_png_file] Error during read_image");

  nb = png_get_rowbytes(png_ptr,info_ptr);
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * I.height);
  for (y=0; y<I.height; y++)
    row_pointers[y] = (png_byte*) malloc(nb);

  png_read_image(png_ptr, row_pointers);

  fprintf(stderr, "read png : width = %d height = %d nb = %d\n",
	  I.width, I.height, nb);
  
  for (y=0; y<I.height; y++)
    for (x=0; x<nb; x++)
      I.c[y*nb + x] = row_pointers[y][x];
  
  fclose(fp);

  for (y=0; y<I.height; y++)
    free(row_pointers[y]);
  free(row_pointers);
  
  return I;
}


void write_png_file(char* file_name, cImage &I)
{
  int nb, x, y;
  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep * row_pointers;

  /* create file */
  FILE *fp = fopen(file_name, "wb");
  if (!fp)
    abort_("[write_png_file] File %s could not be opened for writing", file_name);


  /* initialize stuff */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    abort_("[write_png_file] png_create_write_struct failed");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    abort_("[write_png_file] png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[write_png_file] Error during init_io");

  png_init_io(png_ptr, fp);


  /* write header */
  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[write_png_file] Error during writing header");

  png_set_IHDR(png_ptr, info_ptr, I.width, I.height,
	       I.bit_depth, I.color_type, PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);


  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[write_png_file] Error during writing bytes");

  nb = png_get_rowbytes(png_ptr,info_ptr);
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * I.height);
  for (y=0; y<I.height; y++) {
    row_pointers[y] = (png_byte*) malloc(nb);
    for (x = 0; x<nb; x++)
      row_pointers[y][x] = I.c[y*nb + x];
  }
  png_write_image(png_ptr, row_pointers);


  /* end write */
  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[write_png_file] Error during end of write");

  png_write_end(png_ptr, NULL);

  /* cleanup heap allocation */
  for (y=0; y<I.height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  fclose(fp);
}


