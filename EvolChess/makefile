# http://www.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html

LINK_TARGET = output/evolchess.exe

OBJ_DIR = ./obj

_OBJS =  \
 debug.o \
 engine.o \
 main.o

OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

REBUILDABLES = $(OBJS) $(LINK_TARGET)

clean:
	del $(REBUILDABLES)
	echo Clean done
	
all: $(LINK_TARGET)
	echo All done

$(LINK_TARGET) : $(OBJS)
	g++ -g -o $@ $^

# -MP -MD generates the dependency files .d
$(OBJ_DIR)/%.o : %.cpp
	g++ -MP -MMD -g -o $@ -c $<

# initial hyphen will suppress the error messages that appear when the dependency files do not already exist.
-include $(OBJS:.o=.d)
