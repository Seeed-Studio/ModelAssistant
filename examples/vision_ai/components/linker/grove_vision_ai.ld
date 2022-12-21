/* SYSTEM memory regions indicate where external memory might be located.
   The TCF has no specific knowledge of whether SYSTEM regions contain
   external memory or not.
   CCMWRAP memory regions indicate unusable portions of the address space
   due to CCM memory wrapping into upper addresses beyond its size
*/

#define __ASSEMPLY__
MEMORY {
    ICCM0   : ORIGIN = 0x00000000, LENGTH = 0x00010000
    ICCM1   : ORIGIN = 0x10000000, LENGTH = 0x00050000
    SYSTEM0 : ORIGIN = 0x20010800, LENGTH = (0x2011d650 - 0x20010800)
    VERSION : ORIGIN = 0x2011d650, LENGTH = (0x2011d670 - 0x2011d650)
    DCCM    : ORIGIN = 0x80000000, LENGTH = 0x00040000
    XCCM    : ORIGIN = 0x90000000, LENGTH = 0x00008000
    YCCM    : ORIGIN = 0xa0000000, LENGTH = 0x00008000
#ifdef FLASH_AS_SRAM
    FLASH		: ORIGIN = FLASH_SECTOR_START, LENGTH = 0x00300000
#endif
}

ENTRY(_arc_reset)

SECTIONS {

    .ivt DEFINED (ivtbase_addr) ? ivtbase_addr : 0x10000004 :
    {
        *(.init_vector)
        *(.init_bootstrap)
    } > ICCM1

    .vector : ALIGN(1024)
    {
        . = ALIGN(1024);
        *(.vector)
    } > ICCM1

    /* Read-only sections, merged into text segment: */
    .hash          : { *(.hash)		}
    .dynsym        : { *(.dynsym)		}
    .dynstr        : { *(.dynstr)		}
    .gnu.version   : { *(.gnu.version)	}
    .gnu.version_d   : { *(.gnu.version_d)	}
    .gnu.version_r   : { *(.gnu.version_r)	}
    .rel.init       : { *(.rel.init) }
    .rela.init      : { *(.rela.init) }
    .rel.text       : { *(.rel.text .rel.text.* .rel.gnu.linkonce.t.*) }
    .rela.text      : { *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*) }
    .rel.fini       : { *(.rel.fini) }
    .rela.fini      : { *(.rela.fini) }
    .rel.rodata     : { *(.rel.rodata .rel.rodata.* .rel.gnu.linkonce.r.*) }
    .rela.rodata    : { *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*) }
    .rel.data       : { *(.rel.data .rel.data.* .rel.gnu.linkonce.d.*) }
    .rela.data      : { *(.rela.data .rela.data.* .rela.gnu.linkonce.d.*) }
    .rel.tdata	  : { *(.rel.tdata .rel.tdata.* .rel.gnu.linkonce.td.*) }
    .rela.tdata	  : { *(.rela.tdata .rela.tdata.* .rela.gnu.linkonce.td.*) }
    .rel.tbss	  : { *(.rel.tbss .rel.tbss.* .rel.gnu.linkonce.tb.*) }
    .rela.tbss	  : { *(.rela.tbss .rela.tbss.* .rela.gnu.linkonce.tb.*) }
    .rel.ctors      : { *(.rel.ctors) }
    .rela.ctors     : { *(.rela.ctors) }
    .rel.dtors      : { *(.rel.dtors) }
    .rela.dtors     : { *(.rela.dtors) }
    .rel.got        : { *(.rel.got) }
    .rela.got       : { *(.rela.got) }
    .rel.sdata      : { *(.rel.sdata .rel.sdata.* .rel.gnu.linkonce.s.*) }
    .rela.sdata     : { *(.rela.sdata .rela.sdata.* .rela.gnu.linkonce.s.*) }
    .rel.sbss       : { *(.rel.sbss .rel.sbss.* .rel.gnu.linkonce.sb.*) }
    .rela.sbss      : { *(.rela.sbss .rela.sbss.* .rela.gnu.linkonce.sb.*) }
    .rel.sdata2     : { *(.rel.sdata2 .rel.sdata2.* .rel.gnu.linkonce.s2.*) }
    .rela.sdata2    : { *(.rela.sdata2 .rela.sdata2.* .rela.gnu.linkonce.s2.*) }
    .rel.sbss2      : { *(.rel.sbss2 .rel.sbss2.* .rel.gnu.linkonce.sb2.*) }
    .rela.sbss2     : { *(.rela.sbss2 .rela.sbss2.* .rela.gnu.linkonce.sb2.*) }
    .rel.bss        : { *(.rel.bss .rel.bss.* .rel.gnu.linkonce.b.*) }
    .rela.bss       : { *(.rela.bss .rela.bss.* .rela.gnu.linkonce.b.*) }

    .text           : ALIGN(4)
    {
        . = ALIGN(4);
        PROVIDE (_f_text = .);
        *(TYPE bss)
        KEEP (*(SORT_NONE(.init)))
        /* Start here after reset.  */
        . = ALIGN(4);
        KEEP (*crt0.o(.text.__startup))
        /* Remaining code.  */
        . = ALIGN(4);
        *(.text .stub .text.* .gnu.linkonce.t.*)
        /* .gnu.warning sections are handled specially by elf.em.  */
        *(.gnu.warning)
        . = ALIGN(4);
        *(.mli_lib.*)

        KEEP (*(SORT_NONE(.fini)))
        PROVIDE (__etext = .);
        PROVIDE (_etext = .);
        PROVIDE (etext = .);
        PROVIDE (_e_text = .);
    }  > ICCM1

#ifdef FLASH_AS_SRAM
    #ifdef TFLITE_MICRO_YOLO_COCO
        .rodata : ALIGN(4)
		{
			libtflitemicro_25.a:yolo_coco.o *(.rodata.yolo_coco_tflite)
		}		> FLASH
    #else
		.rodata : ALIGN(4)
		{
			libtflitemicro_24.a:person_detect_model_data.o *(.rodata.g_person_detect_model_data)
		}		> FLASH
    #endif
#endif

    .tensor_arena : ALIGN(16)
    {
        _f_bss_tensor_arena = .;
        *(.bss.tensor_arena)
        _e_bss_tensor_arena = .;
    }  > SYSTEM0

    .cdm_buf : ALIGN(16)
    {
        _f_bss_cdm_buf = .;
        *(.bss.cdm_buf)
        _e_bss_cdm_buf = .;
    }  > SYSTEM0

    .jcr   : ALIGN(4)
    {
        KEEP (*(.jcr))
    } > DCCM

    .eh_frame   : ALIGN(4)
    {
        KEEP (*(.eh_frame))
    } > DCCM

    .gcc_except_table   : ALIGN(4)
    {
        *(.gcc_except_table) *(.gcc_except_table.*)
    } > DCCM

    .plt   : ALIGN(4)
    {
        *(.plt)
    } > DCCM

    .jlitab   : ALIGN(4)
    {
        __JLI_TABLE__ = .;
        jlitab*.o:(.jlitab*) *(.jlitab*)

    } > DCCM

    .read_only_data   : ALIGN(4)
    {
        PROVIDE (_f_rodata = .);
        *(.rodata) *(.rodata.*) *(.gnu.linkonce.r.*)
        *(.rodata1)

        . = ALIGN(4);
        __CTOR_LIST__ = .;
        LONG((__CTOR_END__ - __CTOR_LIST__) / 4 - 2)
        KEEP(*(SORT_BY_NAME(".ctors*")))
        LONG(0)
        __CTOR_END__ = .;

        . = ALIGN(4);
        __init_array_start = .;
        KEEP(*(SORT_BY_NAME(".init_array*")))
        __init_array_end = .;

        . = ALIGN(4);
        __DTOR_LIST__ = .;
        LONG((__DTOR_END__ - __DTOR_LIST__) / 4 - 2)
        KEEP(*(SORT_BY_NAME(".dtors*")))
        LONG(0)
        __DTOR_END__ = .;

        PROVIDE (_e_rodata = .);
    } > SYSTEM0

    .data	  : ALIGN(4)
    {
        /* Start of the data section image in ROM.  */
        PROVIDE (__data_image = .);
        PROVIDE (__data_start = .) ;
        /* --gc-sections will delete empty .data. This leads to wrong start
        addresses for subsequent sections because -Tdata= from the command
        line will have no effect, see PR13697.  Thus, keep .data  */
        KEEP (*(.data))
        *(.data.* .gnu.linkonce.d.*)
        SORT(CONSTRUCTORS)
    	/* gcc uses crtbegin.o to find the start of
        the constructors, so we make sure it is
        first.  Because this is a wildcard, it
        doesn't matter if the user does not
        actually link against crtbegin.o; the
        linker won't look for a file to match a
        wildcard.  The wildcard also means that it
        doesn't matter which directory crtbegin.o
        is in.  */
        KEEP (*crtbegin*.o(.ctors))
        /* We don't want to include the .ctor section from
        from the crtend.o file until after the sorted ctors.
        The .ctor section from the crtend file contains the
        end of ctors marker and it must be last */
        KEEP (*(EXCLUDE_FILE (*crtend*.o ) .ctors))
        KEEP (*(SORT(.ctors.*)))
        KEEP (*(.ctors))
        KEEP (*crtbegin*.o(.dtors))
        KEEP (*(EXCLUDE_FILE (*crtend*.o ) .dtors))
        KEEP (*(SORT(.dtors.*)))
        KEEP (*(.dtors))
    }  > DCCM

    .got            :  ALIGN(4)
    {
        *(.got.plt) *(.got)
    }  > DCCM


    /* We want the small data sections together, so single-instruction offsets
        can access them all, and initialized data all before uninitialized, so
        we can shorten the on-disk segment size.  */
    .sdata          : ALIGN(8)
    {
        __SDATA_BEGIN__ = . ;
        _f_sdata = .;
        *(.sdata .sdata.* .gnu.linkonce.s.*)
        _edata  =  .;
        PROVIDE (edata = .);

        *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
        /* Global data not cleared after reset.  */
        *(.noinit*)
        . = ALIGN(32 / 8);
    }  > DCCM

    .sbss_all           : ALIGN(4)
    {
        PROVIDE (__sbss_start = .);
        PROVIDE (___sbss_start = .);
        *(.dynsbss)
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        *(.scommon)
        PROVIDE (__sbss_end = .);
        PROVIDE (___sbss_end = .);
        *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*)
    }  > DCCM


    .bss_all           : ALIGN(4)
    {
        PROVIDE (_f_bss = .) ;
        *(.dynbss)
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        /* Align here to ensure that the .bss section occupies space up to
            _end.  Align after .bss to ensure correct alignment even if the
            .bss section disappears because there are no input sections.  */
        . = ALIGN(32 / 8);
        _end = .;
        PROVIDE (end = .);

        PROVIDE (_e_bss = .) ;
    }  > SYSTEM0

    .stack : ALIGN(8)
    {
        __start_stack = . ;
        _f_stack = .;
        . = . + _STACKSIZE;
        _e_stack = .;
        __end_stack = . ;
    } > DCCM

    .heap : ALIGN(8)
    {
        __start_heap = . ;
        _f_heap = .;
        . = . + _HEAPSIZE;
        _e_heap = .;
        __end_heap = . ;
    } > DCCM

    .Zdata : ALIGN(16)
    {
        . = ALIGN(16);
        *(.Zdata*)
        /* PROVIDE (__start_heap = .); */
        /* PROVIDE (_f_heap = .); */
    }  > DCCM

    .Xdata : ALIGN(4)
    {
        *(.Xdata*)
    }  > XCCM

    .Ydata : ALIGN(4)
    {
        *(.Ydata*)
    }  > YCCM

    .ver : ALIGN(16)
    {
        *(.data*)
    }  > VERSION

    /* PROVIDE (__stack_top = (ORIGIN (DCCM) + LENGTH (DCCM) - 1) & -4); */
    /* PROVIDE (__end_heap = ORIGIN (DCCM) + LENGTH (DCCM) - 1); */

    /* PROVIDE (_e_stack = (ORIGIN (DCCM) + LENGTH (DCCM) - 8192) & -4); */
    /* PROVIDE (_e_heap = __end_heap); */
    PROVIDE (_load_addr_text = _f_text);
    PROVIDE (_load_addr_rodata = _f_rodata);

    /* Stabs debugging sections.  */
    .stab          0 : { *(.stab) }
    .stabstr       0 : { *(.stabstr) }
    .stab.excl     0 : { *(.stab.excl) }
    .stab.exclstr  0 : { *(.stab.exclstr) }
    .stab.index    0 : { *(.stab.index) }
    .stab.indexstr 0 : { *(.stab.indexstr) }
    .comment       0 : { *(.comment) }
    .note.gnu.build-id : { *(.note.gnu.build-id) }
    /* DWARF debug sections.
        Symbols in the DWARF debugging sections are relative to the beginning
        of the section so we begin them at 0.  */
    /* DWARF 1 */
    .debug          0 : { *(.debug) }
    .line           0 : { *(.line) }
    /* GNU DWARF 1 extensions */
    .debug_srcinfo  0 : { *(.debug_srcinfo) }
    .debug_sfnames  0 : { *(.debug_sfnames) }
    /* DWARF 1.1 and DWARF 2 */
    .debug_aranges  0 : { *(.debug_aranges) }
    .debug_pubnames 0 : { *(.debug_pubnames) }
    /* DWARF 2 */
    .debug_info     0 : { *(.debug_info .gnu.linkonce.wi.*) }
    .debug_abbrev   0 : { *(.debug_abbrev) }
    .debug_line     0 : { *(.debug_line .debug_line.* .debug_line_end) }
    .debug_frame    0 : { *(.debug_frame) }
    .debug_str      0 : { *(.debug_str) }
    .debug_loc      0 : { *(.debug_loc) }
    .debug_macinfo  0 : { *(.debug_macinfo) }
    /* SGI/MIPS DWARF 2 extensions */
    .debug_weaknames 0 : { *(.debug_weaknames) }
    .debug_funcnames 0 : { *(.debug_funcnames) }
    .debug_typenames 0 : { *(.debug_typenames) }
    .debug_varnames  0 : { *(.debug_varnames) }
    /* DWARF 3 */
    .debug_pubtypes 0 : { *(.debug_pubtypes) }
    .debug_ranges   0 : { *(.debug_ranges) }
    /* DWARF Extension.  */
    .debug_macro    0 : { *(.debug_macro) }
    .debug_addr     0 : { *(.debug_addr) }
    /* ARC Extension Sections */
    .arcextmap	  0 : { *(.arcextmap.*) }
}


