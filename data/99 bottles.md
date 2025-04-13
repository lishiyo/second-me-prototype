# 99 bottles
#coding

99 Bottles - Sandi Metz

### Chapter 2

[Clean Coder Blog](https://blog.cleancoder.com/uncle-bob/2013/05/27/TheTransformationPriorityPremise.html)
> Refactorings have counterparts called*Transformations*. Refactorings are simple operations that change the structure of code without changing it’s behavior.*Transformations*are simple operations that change the behavior of code.

Transform from specific to generic.
Return the literal answer
Return a variable (const -> variable

Conditional to a interpolating any variable given -> reduce complexity
constant -> scalar (“replacing a constant with a var/arg”)
unconditional -> if (“splitting the execution path”)

4 basic cases: 3-97, 2, 1, 0
pluralization is not the correct abstraction
optimize for understandability, not changeability
tolerate duplication if that helps reveal the abstraction
don’t create an abstraction until you’re sure what it really is

`if/elseif` implies that each condition varies in a meaningful way
`case` implies that conditions are mostly the same, outside one value

quickly maximize number of whole examples before extracting abstractions from their parts 

“Fake it” - write just enough code to pass the tests
Obvious Implementation
Triangulate - write several tests at once, then try to write one piece of code which makes all of them pass

Write tests that *confirm* code without any knowledge of how

### Chapter 3

New requirement - exactly how it should change
alter 99 bottles to output “1 six-pack” where it says “6 bottles”

open/closed principle - “open for extension, closed for modification”
can you meet the new requirement w/o modifying existing code?

easiest way to find code smells - what don’t you like about this code?
fight one code smell at a time
duplicated code -> refactor to remove
refactor = alter the arrangement of code w/o changing its behavior
tests = safety blanket
if tests start breaking, improve them first *then* refactor

Flocking Rules allow for abstractions to appear:
* Select the things that are most alike
* Find the smallest diff b/t them
* Make the simplest change that will remove the difference:
  * parse the new code
  * parse and execute it
  * parse, execute, and use its result
  * deleted unused code

You do NOT have to identify the underlying abstractions before refactoring. Write the code dictated by the rules, then the abstractions will follow.

> The “bus factor” is the minimum number of team members that have to suddenly disappear from a project before the project stalls due to lack of knowledgeable or competent personnel.

the bottle/bottles abstraction is not pluralization, but `container`
1 bottle, 1 six-pack, n bottles
1 carafe, 1 glass etc

### Chapter 4

