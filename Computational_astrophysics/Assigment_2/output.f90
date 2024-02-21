module out
    implicit none
    integer:: n_file = 0

contains
    subroutine output(x,y,t)
        real, intent(inout):: x,y, t
        character(len=20):: file_name

        write(file_name,"(a,i5.5)") 'output_', n_file
        open(2, file = file_name, status = 'replace', action ='write')
        n_file = n_file +1
        write(2,*) '# x, y, t'
        write(2,*) t

        write(2,*) x,y

        close(2)
    end subroutine output
end module
