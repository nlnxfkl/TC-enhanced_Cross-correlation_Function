! SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
! list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
! this list of conditions and the following disclaimer in the documentation
! and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
! contributors may be used to endorse or promote products derived from
! this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!

! #define USE_CUBLAS

#ifndef USE_CUBLAS
#define USE_TF32
#endif
      program TC_enhanced_cross_correlation_function
      implicit none
      !----------------------------------------------------------------!
      ! the number of timesteps of template wave
      integer*4,parameter:: nm=256
      ! integer*4,parameter:: nm=512
      ! integer*4,parameter:: nm=1024
      ! integer*4,parameter:: nm=2048
      ! integer*4,parameter:: nm=4096
      ! integer*4,parameter:: nm=8192
      ! integer*4,parameter:: nm=180000
      ! integer*4,parameter:: nm=179968
      ! integer*4,parameter:: nm=20500
      !----------------------------------------------------------------!
      integer*4 ntori,nwori,nc,nt,nw
      real*8 eps

      open(10,file='./data/parameter_setting.dat',status='unknown')
      read(10,*) ! the number of timesteps of observation data
      read(10,*) ntori
      read(10,*) ! the number of template waves
      read(10,*) nwori
      read(10,*) ! tolerance to detect a event
      read(10,*) eps
      close(10)

      write(*,*) "the number of timesteps of template wave: ", nm
      write(*,*) "the number of timesteps of observation data: ", ntori

#ifdef USE_CUBLAS
      write(*,*) 'use cuBLAS'
#endif
#ifndef USE_CUBLAS
#ifdef USE_TF32
      write(*,*) 'use TF32'
#else
      write(*,*) 'use FP16'
#endif
#endif

      nt = int((ntori-1)/(NTENSOR*NBLOCK)+1)*(NTENSOR*NBLOCK)
      nw = int((nwori-1)/NTENSOR+1)*NTENSOR
      nc = nt-nm

      write(*,*) "nt, nw, nc: ",nt,nw,nc

      call main_prepare(ntori,nwori,nt,nw,nc,nm,eps)

      end program

c----------------------------------------------------------------------
      subroutine main_prepare(ntori,nwori,nt,nw,nc,nm,eps)
      implicit none
      integer*4 ntori,nwori,nt,nw,nc,nm
      real*8 eps
      real*8 obs(nt),tpl(nm,NTENSOR),tplori(nm,NTENSOR)
      real*8 obsco(nt)
      real*4 tplscale(nm/NTENSOR,NTENSOR)
      real*8 co(nt,NTENSOR)
      real*8 tmp,maxtmp,mintmp
      real*8 t_cpu,t_gpu
      character*100 filename,listname

#ifndef USE_CUBLAS
      real*4,device:: obs_s_d(nt)
#endif

      integer*4 it,i,j,ic
      integer*4 i_tplbase,i_tpl,i_16
      integer*4 istat,ierr

      tpl(:,:)=0d0
      obs(:)=0d0
      t_cpu=0d0
      t_gpu=0d0

      !----------------------------------------------------------------!
      open(11,file='./data/obs_file.dat',status='unknown')
      ! open(11,file='../data/obs_file.dat.actual',status='unknown')
      !----------------------------------------------------------------!
      read(11,*) !filename for observation data
      read(11,*) filename
      close(11)

      open(41,file=trim(filename),status='old')
      do it=1,ntori
      read(41,*) tmp,obs(it)
      enddo
      close(41)

      tmp=0d0
      maxtmp=0d0
      mintmp=1d5
      do i=1,nt
        tmp=tmp+abs(obs(i))
        maxtmp=max(maxtmp,abs(obs(i)))
        mintmp=min(mintmp,abs(obs(i)))
      enddo
      tmp=tmp/nt

      do i=1,nt
        obs(i)=obs(i)/tmp
      enddo

      tmp=0d0
      do it=1,nm
        tmp=tmp+obs(it)**2
      enddo
      obsco(1)=sqrt(tmp)
      write(*,*) 'reach'
      do ic=1,nc
        tmp=tmp-obs(ic)**2
        tmp=tmp+obs(nm+ic)**2
        obsco(1+ic)=sqrt(tmp)
      enddo

#ifndef USE_CUBLAS
      obs_s_d=obs
#endif

      write(*,'(a24)') "observation data loaded."

      !----------------------------------------------------------------!
      open(12,file='./data/tpl_file.dat',status='unknown')
      ! open(12,file='../data/tpl_file.dat.512',status='unknown')
      ! open(12,file='../data/tpl_file.dat.1024',status='unknown')
      ! open(12,file='../data/tpl_file.dat.2048',status='unknown')
      ! open(12,file='../data/tpl_file.dat.4096',status='unknown')
      ! open(12,file='../data/tpl_file.dat.8192',status='unknown')
      ! open(12,file='../data/tpl_file.dat.179968',status='unknown')
      ! open(12,file='../data/tpl_file.dat.actual',status='unknown')
      !----------------------------------------------------------------!
      read(12,*) !filename for template data

      open(13,file='res.csv',status='unknown')

      do i_tplbase=1,nw,NTENSOR
        write(*,'(a27,i7.6,a3,i7.6)')
     &   "cross-correlation function for wave",
     &   i_tplbase,'to',min(i_tplbase+NTENSOR-1,nwori)
        tplori(:,:)=0d0
        tpl(:,:)=0d0

        do i_tpl=i_tplbase,min(i_tplbase+NTENSOR-1,nwori)
        i_16=i_tpl-i_tplbase+1
        read(12,*) filename
        open(41,file=trim(filename),status='old')
        do it=1,nm
          read(41,*) tmp,tplori(it,i_16)
        enddo
        close(41)

        tmp=0d0
        maxtmp=0d0
        mintmp=1d5
        do i=1,nm,NTENSOR
          maxtmp=0d0
          do j=i,i+NTENSOR-1
            maxtmp=max(maxtmp,abs(tplori(j,i_16)))
          enddo
          maxtmp=maxtmp/dble(NTENSOR)
          if (maxtmp<1d-5) maxtmp=1d0
          tplscale(i/NTENSOR+1,i_16)=maxtmp
          do j=i,i+NTENSOR-1
            tpl(j,i_16)=tplori(j,i_16)/maxtmp
          enddo
        enddo
        enddo ! i_tpl

#ifndef USE_CUBLAS
        call main_compute(
     &   nc,nt,nm,nw,obs,obsco,tplori,tpl,tplscale,co,obs_s_d,
     &   t_cpu,t_gpu)
#else
        call main_compute(
     &   nc,nt,nm,nw,obs,obsco,tplori,tpl,tplscale,co,
     &   t_cpu,t_gpu)
#endif

        do i_tpl=i_tplbase,min(i_tplbase+NTENSOR-1,nwori)
          i_16=i_tpl-i_tplbase+1
          do i=1,ntori-nm+1
            if(co(i,i_16).gt.eps)then
              write(13,*) i,',',i_tpl,',',co(i,i_16)
            endif
          enddo
        enddo

      enddo !i_tplbase

      write(*,'(a22)') "computation completed."
#ifdef VERIFICATION
      write(6,'(a23,a2,e14.7,a4)')
     & 'Computation time on CPU',':',t_cpu,'sec'
#endif
      write(6,'(a23,a2,e14.7,a4)')
     & 'Computation time on GPU',':',t_gpu,'sec'

      close(12)
      close(13)

      return
      end

c-----------------------------------------------------------------------
#ifndef USE_CUBLAS
      subroutine main_compute(
     & nc,nt,nm,nw,obs,obsco,tplori,tpl,tplscale,co,obs_s_d,
     & t_cpu,t_gpu)
#else
      subroutine main_compute(
     & nc,nt,nm,nw,obs,obsco,tplori,tpl,tplscale,co,
     & t_cpu,t_gpu)
#endif
      use cudafor
      integer*4 nc,nt,nm,nw
      real*8 obs(nt),tpl(nm,NTENSOR),tplori(nm,NTENSOR),tplco(NTENSOR)
      real*8 obsco(nt),tmp,co(nt,NTENSOR),obsl(nm)
      real*4 obs_s(nt),tpl_s(nm,NTENSOR),co_s(nt,NTENSOR)
#ifdef USE_CUBLAS
      real*4 co_s_t(NTENSOR,nt)
#endif
      real*4 tpl_h(nm,NTENSOR)
      real*8 co_ori(nt,NTENSOR)
      real*4 tplscale(nm/NTENSOR,NTENSOR)

      real*8 max_abs_error
      !device memory
#ifndef USE_CUBLAS
      integer*4,device:: nt_d,nm_d,nc_d
      real*4,device:: obs_s_d(nt),co_s_d(nt,NTENSOR)
      real*4,device:: tpl_h_d(nm,NTENSOR),tplscale_d(nm/NTENSOR,NTENSOR)
#endif

      integer*4 block_size,num_warps
      integer*4 grid_size
      integer*4 n_iters
      real*8 t1,t2
      real*8 t_cpu,t_gpu

      external:: compdef_gputc

#ifdef USE_CUBLAS
      obs_s = obs
#endif

      tplco(:)=0d0
      do i=1,NTENSOR
        do it=1,nm
          tplco(i)=tplco(i)+tplori(it,i)**2
        enddo
        tplco(i)=sqrt(tplco(i))
      enddo

#ifdef VERIFICATION
      write(*,*) 'cpu calculation start'
      write(*,*) '...'
      call cpu_time(t1)
      do ib=1,nc+1
        ie=ib+nm-1
        do it=ib,ie
          obsl(it-ib+1)=obs(it)
        enddo
        do i=1,NTENSOR
          tmp=0d0
          do it=1,nm
            tmp=tmp+obsl(it)*tplori(it,i)
          enddo
          co(ib,i)=tmp
        enddo
      enddo
      call cpu_time(t2)
      t_cpu=t_cpu+(t2-t1)
      co_ori(:,:)=co(:,:)
#endif
      itmp=nm*NTENSOR
      !----------------------------------------------------------------!
#ifdef USE_CUBLAS
      call convert_fp32(itmp,tplori,tpl_h)
      write(*,*) 'check: convert_fp32() was called.'
#else
#ifndef USE_TF32
      call convert_fp16(itmp,tpl,tpl_h)
      write(*,*) 'check: convert_fp16() was called.'
      ! call convert_bf16(itmp,tplori,tpl_h)
      ! call convert_bf16(itmp,tpl,tpl_h)
#else
      call convert_fp32(itmp,tplori,tpl_h)
      write(*,*) 'check: convert_fp32() was called.'
#endif
#endif
      !----------------------------------------------------------------!
#ifndef USE_CUBLAS
      nt_d=nt
      nm_d=nm
      nc_d=nc
      co_s_d(:,:)=0d0

      tpl_h_d(:,:)=tpl_h(:,:)
      tplscale_d(:,:)=tplscale(:,:)
#endif

      istat=cudaDeviceSynchronize()

      write(*,*) 'gpu calculation start'
      write(*,*) '...'
#ifndef USE_CUBLAS
      call cpu_time(t1)
      block_size = 128
      nwarps = block_size / 32

#ifdef VERIFICATION
      n_iters = 1
      call call_compdef_gputc
     & (nt_d,nm_d,nc_d,obs_s_d,tpl_h_d,co_s_d)
#else
      n_iters = 1000
      do i=1,n_iters
        call call_compdef_gputc
     &   (nt_d,nm_d,nc_d,obs_s_d,tpl_h_d,co_s_d)
      enddo
#endif
      istat=cudadevicesynchronize()
      call cpu_time(t2)
      write(*,*) 'gpu calculation end'
      t_gpu=t_gpu+(t2-t1)/n_iters
#else
      call cublas_wrapper(nt,nm,nc,obs_s,transpose(tpl_h),co_s_t,t_gpu)
#endif

#ifndef USE_CUBLAS
      co_s(:,:)=co_s_d(:,:)
#else
      co_s = transpose(co_s_t)
#endif

      do i=1,NTENSOR
      do ic=1,nc+1
        co(ic,i)=co_s(ic,i)/obsco(ic)/tplco(i)
      enddo
      enddo

#ifdef VERIFICATION
      do i=1,NTENSOR
      do ic=1,nc+1
        co_ori(ic,i)=co_ori(ic,i)/obsco(ic)/tplco(i)
      enddo
      enddo

      ! check the result
      do ic=1,1000
        if (ic>nc+1) exit
        write(*,*) co(ic,1),co_ori(ic,1),abs(co(ic,1)-co_ori(ic,1))
      enddo

      max_abs_error = 0
      do i=1,NTENSOR
      do ic=1,nc+1
        max_abs_error = max(max_abs_error, abs(co(ic,i)-co_ori(ic,i)))
        tmp=abs((co(ic,i)-co_ori(ic,i))/co_ori(ic,i))
        if (tmp>0.1 .and. co_ori(ic,i)>0.1) then
        write(*,'(a40,i10,i10,f13.6,2e14.3,i4)')
     &  'time,template,error perc.,co,co_ori,id',
     & ic,i,tmp*100,co(ic,i),co_ori(ic,i)
        endif
      enddo
      enddo
      write(*,*) 'maximum absolute error to FP64: ',max_abs_error
#endif
      return
      end subroutine
